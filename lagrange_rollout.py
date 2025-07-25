import os
import numpy as np
import torch
import h5py
from huggingface_hub import hf_hub_download
import gymnasium

from humenv import STANDARD_TASKS, make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper
from metamotivo.buffers.buffers import DictBuffer
from utils import set_seed, suggest_constraint_range
import mediapy as media

os.environ["OMP_NUM_THREADS"] = "1"

METHOD = "bisection" # gradient_descent, baseline, bisection

if __name__ == "__main__":
    local_dir = "metamotivo-S-1-datasets"
    dataset = "buffer_inference_500000.hdf5"
    buffer_path = hf_hub_download(
        repo_id="facebook/metamotivo-S-1",
        filename=f"data/{dataset}",
        repo_type="model",
        local_dir=local_dir,
    )
    hf = h5py.File(buffer_path, "r")
    data = {k: v[:] for k, v in hf.items()}
    buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
    buffer.extend(data)

    model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", device="cpu")
    rew_model = RewardWrapper(
        model=model,
        inference_dataset=buffer,
        num_samples_per_inference=100_000,
        inference_function="reward_wr_inference",
        max_workers=40,
        process_executor=True,
        process_context="forkserver"
    )

    body_part = "L_Hip" # Check body_names.txt for all body parts
    kind = "vel" # "pos", "rot", "vel" or "ang"
    output_file = f"lagrange_output_{body_part}_{kind} ({METHOD}).txt"

    with open(output_file, "w") as f:
        for task in STANDARD_TASKS:
            print(f"\n🎯 Task: {task}")
            f.write(f"\n🎯 Task: {task}\n")
            z = rew_model.reward_inference(task)

            # get sample range and specific dimensions
            dim_idx_set, lo, hi = suggest_constraint_range(task=task, body=body_part, kind=kind)
            specific_dimensions = list(dim_idx_set)
            sample_range = (lo, hi)

            task_rewards = []
            task_costs = []
            q_c_values = []
            q_c_initials = []

            for seed in range(5):
                set_seed(seed)
                env, _ = make_humenv(num_envs=1, task=task, state_init="Default", seed=seed,
                                     wrappers=[gymnasium.wrappers.FlattenObservation])
                observation, info = env.reset(seed=seed)

                # Lagrange optimization
                observation_c_z = observation.copy()
                for dim in specific_dimensions:
                    observation_c_z[dim] = np.random.uniform(sample_range[0], sample_range[1])
                observation_c_z_tensor = torch.tensor(observation_c_z.reshape(1, -1), dtype=torch.float32)
                modified_values = observation_c_z_tensor[:, specific_dimensions]
                reward_term = 10 * (modified_values - sample_range[0])
                reward_term = torch.sum(reward_term, dim=1, keepdim=True)
                c_z = model.reward_inference(observation_c_z_tensor, reward_term)

                eta = -300
                lr = 0.0001 
                obs_torch = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=z.device)
                
                # Define lagrange_min, and lagrange_max for bisection method
                lambda_min_t, lambda_max_t = 0.0, 1.0
                lambda_t = (lambda_min_t + lambda_max_t) / 2.0
                
                # Define threshold and step size for bisection and baseline methods
                threshold = 0.01
                step_size = 0.1

                # 1) make a learnable lambda_t (if method is gradient_descent)
                requires_grad = (METHOD == "gradient_descent")
                lambda_t = torch.tensor(lambda_t, dtype=torch.float32, device=z.device,
                                        requires_grad=requires_grad)
                                    
                lambda_min_t = torch.tensor(lambda_min_t, dtype=torch.float32, device=z.device,
                                        requires_grad=False)
                lambda_max_t = torch.tensor(lambda_max_t, dtype=torch.float32, device=z.device,
                                        requires_grad=False)

                # 2) turn OFF the global no_grad for the critic call
                model.eval()                 # keep eval mode (no dropout, etc.)

                critic_net = model._critic    # raw nn.Module (not wrapped)

                for step in range(100):
                    if lambda_t.grad is not None and METHOD == "gradient_descent":
                        lambda_t.grad.zero_()

                    # --- build Z_{λ,c} ---
                    Z_lambda_c = z - lambda_t * c_z          # this depends on λ

                    # action can still come from the helper (we don't need its grad)
                    with torch.no_grad():
                        a = model.act(obs=obs_torch, z=Z_lambda_c).ravel()

                    # 3) call the *raw* critic so autograd can see λ
                    Q = critic_net(
                            obs_torch,
                            Z_lambda_c.unsqueeze(0),
                            a.unsqueeze(0).unsqueeze(0)
                        ).squeeze()

                    # Store initial Q_c values
                    if step == 0: q_c_initials.append(Q.mean().item())

                    if (METHOD == "gradient_descent"):
                        loss = 0.5 * (Q.mean() - eta).pow(2)    # scalar

                        # 4) backward through λ
                        loss.backward()                          # fills lambda_t.grad

                        # 5) manual SGD on λ (parameters stay frozen)
                        with torch.no_grad():
                            lambda_t -= lr * lambda_t.grad

                        # print(f"step {step:3d} | λ = {lambda_t.item():.4f} | Q = {Q[0].item():.4f} | loss = {loss.item():.4f}")
                    elif (METHOD == "bisection"):
                        if (abs(Q.mean() - eta) < threshold):
                            break
                        elif (Q.mean() > eta):
                            lambda_max_t = lambda_t
                        elif (Q.mean() < eta):
                            lambda_min_t = lambda_t
                        lambda_t = 0.5 * (lambda_min_t + lambda_max_t)
                    else: # baseline
                        sign = torch.sign(Q.mean() - eta)        # +1 if Q>η, -1 if Q<η
                        # heuristic: move against the sign to push Q toward η
                        lambda_t.add_(sign * step_size)
                        if abs(Q.mean() - eta) < threshold:
                            break

                # Final rollout
                observation, _ = env.reset()
                done = False
                total_reward = 0.0
                total_cost = 0.0
                q_c = 0.0
                n_steps = 0

                frames = [env.render()]
                while not done:
                    Z_lambda_c = z - lambda_t.item() * c_z
                    obs = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=z.device)
                    action = rew_model.act(obs, Z_lambda_c).ravel()
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    for dim in specific_dimensions:
                        if observation[dim] > sample_range[0]:
                            total_cost += 10 * (observation[dim] - sample_range[0])
                    frames.append(env.render())
                    done = bool(terminated or truncated)

                    # store Q_c for first time
                    if n_steps == 0:
                        action = torch.tensor(action, dtype=torch.float32, device=z.device)
                        q_c = rew_model.critic(obs, Z_lambda_c.unsqueeze(0), action.unsqueeze(0).unsqueeze(0)).squeeze().mean().item()
                    n_steps += 1

                task_rewards.append(total_reward)
                task_costs.append(total_cost)
                q_c_values.append(q_c)

                # save video
                video_dir = "videos"
                os.makedirs(video_dir, exist_ok=True)
                range_str = f"{lo:.3f}_{hi:.3f}".replace('.', 'p')
                dims_str = f"dims{'_'.join(map(str, specific_dimensions))}"
                video_filename = f"{task.replace('/', '_')}_seed{seed}_{range_str}_{dims_str}_eta_{eta}_{body_part}_{kind}_lagrange.mp4"
                video_path = os.path.join(video_dir, video_filename)
                # if (seed == 0):
                #     media.write_video(video_path, frames, fps=30)

            reward_mean = np.mean(task_rewards)
            reward_std = np.std(task_rewards)
            cost_mean = np.mean(task_costs)
            cost_std = np.std(task_costs)
            q_c_mean = np.mean(q_c_values)
            q_c_std = np.std(q_c_values)
            q_c_initial_mean = np.mean(q_c_initials)
            q_c_initial_std = np.std(q_c_initials)

            result = (
                f"✅ Reward: {reward_mean:.2f} ± {reward_std:.2f}\n"
                f"⚠️ Cost:   {cost_mean:.2f} ± {cost_std:.2f}\n"
                f"🔍 Q_c Initial: {q_c_initial_mean:.2f} ± {q_c_initial_std:.2f}\n"
                f"💰 Q_c Final:    {q_c_mean:.2f} ± {q_c_std:.2f}\n"
            )
            print(result)
            f.write(result)

    print(f"\n📄 All results saved to {output_file}")
