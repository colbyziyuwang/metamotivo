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
    output_file = f"IIP_output_{body_part}_{kind} ({METHOD}).txt"

    with open(output_file, "w") as f:
        for task in STANDARD_TASKS:
            print(f"\n🎯 Task: {task}")
            f.write(f"\n🎯 Task: {task}\n")

            # 1) Get z_star vector
            z_star = rew_model.reward_inference(task)

            # get sample range and specific dimensions
            dim_idx_set, lo, hi = suggest_constraint_range(task=task, body=body_part, kind=kind)
            specific_dimensions = list(dim_idx_set)
            sample_range = (lo, hi)

            task_rewards = []

            for seed in range(5):
                set_seed(seed)
                env, _ = make_humenv(num_envs=1, task=task, state_init="Default", seed=seed,
                                     wrappers=[gymnasium.wrappers.FlattenObservation])
                observation, info = env.reset(seed=seed)

                # 2) Pertube the obervation vector
                observation_IIP_z = observation.copy()
                for dim in specific_dimensions:
                    observation_IIP_z[dim] = np.random.uniform(sample_range[0], sample_range[1])
                observation_IIP_z_tensor = torch.tensor(observation_IIP_z.reshape(1, -1), dtype=torch.float32)

                # 3) Get IIP_z vector (L2 distance)
                back_embed = model.backward_map(observation_IIP_z_tensor)
                reward_term = (back_embed - z_star).pow(2).sum(dim=-1, keepdim=True).sqrt()
                IIP_z = model.reward_inference(observation_IIP_z_tensor, reward=reward_term)
                
                # Define lagrange_min, and lagrange_max for bisection method
                lambda_min_t, lambda_max_t = 0.0, 1.0
                lambda_t = (lambda_min_t + lambda_max_t) / 2.0
                
                model.eval()

                for step in range(100):
                    # 4) Update skill vector
                    z = z_star - lambda_t * IIP_z   # this depends on λ

                    # 5) Begin rollout and collect history of (s, r)
                    n_steps = 10
                    hist = []

                    observation, _ = env.reset()
                    for _ in range(n_steps):
                        obs = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=z.device)
                        action = rew_model.act(obs, z).ravel()
                        observation, reward, terminated, truncated, info = env.step(action)
                        hist.append((obs, reward))
                    
                    # 6) Sample a tuple of (s, r) from history
                    sample_idx = np.random.randint(len(hist))
                    obs_sample, reward_sample = hist[sample_idx]

                    # 7) Compute distance from IIP_z:
                    back_embed = model.backward_map(obs_sample)
                    dist = (back_embed * reward_sample - IIP_z).pow(2).sum(dim=-1, keepdim=True).sqrt()
                    threshold = 100 # want to maximize dist
                    eps = 0.01

                    if (abs(dist - threshold) < eps):
                        break
                    elif (dist > threshold):
                        lambda_min_t = lambda_t
                    elif (dist < threshold):
                        lambda_max_t = lambda_t
                    lambda_t = 0.5 * (lambda_min_t + lambda_max_t)

                # Final rollout
                observation, _ = env.reset()
                done = False
                total_reward = 0.0
 
                frames = [env.render()]
                while not done:
                    Z_lambda_c = z_star - lambda_t * IIP_z
                    obs = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=z.device)
                    action = rew_model.act(obs, Z_lambda_c).ravel()
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    frames.append(env.render())
                    done = bool(terminated or truncated)

                task_rewards.append(total_reward)

                # save video
                video_dir = "videos"
                os.makedirs(video_dir, exist_ok=True)
                range_str = f"{lo:.3f}_{hi:.3f}".replace('.', 'p')
                dims_str = f"dims{'_'.join(map(str, specific_dimensions))}"
                video_filename = f"{task.replace('/', '_')}_seed{seed}_{range_str}_{dims_str}_{body_part}_{kind}_IIP.mp4"
                video_path = os.path.join(video_dir, video_filename)
                #if (seed == 0):
                     #media.write_video(video_path, frames, fps=30)

            reward_mean = np.mean(task_rewards)
            reward_std = np.std(task_rewards)
            result = (
                f"✅ Reward: {reward_mean:.2f} ± {reward_std:.2f}\n"
            )
            print(result)
            f.write(result)

    print(f"\n📄 All results saved to {output_file}")
