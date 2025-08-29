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
import random

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

    output_file = f"IIP_output_({METHOD}).txt"

    with open(output_file, "w") as f:
        for task1 in STANDARD_TASKS:
            print(f"\n🎯 Task: {task1}")
            f.write(f"\n🎯 Task: {task1}\n")

            # Sample a different task
            alt_tasks = [t for t in STANDARD_TASKS if t != task1]
            task2 = random.choice(alt_tasks)
            print(f"\n🎯 Task: {task2}")
            f.write(f"\n🎯 Task: {task2}\n")

            # 1) Get z_vector for both tasks
            z1 = rew_model.reward_inference(task1)
            z2 = rew_model.reward_inference(task2)

            task_rewards = []

            for seed in range(5):
                set_seed(seed)
                env1, _ = make_humenv(num_envs=1, task=task1, state_init="Default", seed=seed,
                                     wrappers=[gymnasium.wrappers.FlattenObservation])
                env2, _ = make_humenv(num_envs=1, task=task2, state_init="Default", seed=seed,
                                     wrappers=[gymnasium.wrappers.FlattenObservation])

                # 3) Get IIP_z vector (L2 distance)
                lambda_min = -10
                lambda_max = 10
                lambda_t = (lambda_min + lambda_max) / 2.0
                
                model.eval()

                for step in range(100):
                    # 4) Update skill vector
                    IIP_z = z1 - lambda_t * z2  # this depends on λ

                    # 5) Begin rollout and collect history of (s, r)
                    n_steps = 10
                    reward_total = 0.0

                    observation, _ = env2.reset()
                    for _ in range(n_steps):
                        obs = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=IIP_z.device)
                        action = rew_model.act(obs, IIP_z).ravel()
                        observation, reward, terminated, truncated, info = env2.step(action)
                        reward_total += reward
                    reward_total = reward_total / n_steps

                    # Set a threshold reward
                    threshold = 10
                    eps = 0.1

                    if (abs(reward_total - threshold) < eps):
                        break
                    elif (reward_total > threshold): # increase lambda
                        lambda_min = lambda_t
                    elif (reward_total < threshold): # decrease lambda
                        lambda_max = lambda_t
                    lambda_t = 0.5 * (lambda_min + lambda_max)

                # Final rollout in env1
                observation, _ = env1.reset()
                done = False
                total_reward = 0.0
 
                frames = [env1.render()]
                while not done:
                    IIP_z = z1 - lambda_t * z2
                    obs = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=IIP_z.device)
                    action = rew_model.act(obs, IIP_z).ravel()
                    observation, reward, terminated, truncated, info = env1.step(action)
                    total_reward += reward
                    frames.append(env1.render())
                    done = bool(terminated or truncated)

                task_rewards.append(total_reward)

                # save video
                video_dir = "videos"
                os.makedirs(video_dir, exist_ok=True)
                video_filename = f"{task1}_to_{task2}_IIP.mp4"
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
