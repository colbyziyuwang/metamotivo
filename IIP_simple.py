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
import cv2

os.environ["OMP_NUM_THREADS"] = "1"

METHOD = "bisection" # gradient_descent, baseline, bisection

# Utility function to add text overlay to each frame
def add_text_to_frames(frames, text):
    output_frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame in frames:
        frame_with_text = frame.copy()
        cv2.putText(frame_with_text, text, (10, 40), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        output_frames.append(frame_with_text)
    return output_frames

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

    task_combos = [("headstand", "rotate-x--5-0.8")] # 5
    # [("headstand", "rotate-z--5-0.8")] # 5
    # [("jump-2", "raisearms-m-m")] # 5 (best demo)
    # [("move-ego--90-2", "raisearms-h-l")] # 11
    # [("jump-2", "raisearms-h-h")] # 10
    # [("split-1", "sitonground")] # 5

    with open(output_file, "w") as f:
        for task_comb in task_combos:
            task1, task2 = task_comb
            print(f"\n🎯 Task: {task1}")
            f.write(f"\n🎯 Task: {task1}\n")

            # Sample a different task
            print(f"\n🎯 Task: {task2}")
            f.write(f"\n🎯 Task: {task2}\n")

            # 1) Get z_vector for both tasks
            z1 = rew_model.reward_inference(task1)
            z2 = rew_model.reward_inference(task2)

            task_rewards = []

            for seed in range(1):
                set_seed(seed)
                env1, _ = make_humenv(num_envs=1, task=task1, state_init="Default", seed=seed,
                                     wrappers=[gymnasium.wrappers.FlattenObservation])
                env2, _ = make_humenv(num_envs=1, task=task2, state_init="Default", seed=seed,
                                     wrappers=[gymnasium.wrappers.FlattenObservation])

                # 3) Get IIP_z vector (L2 distance)
                lambda_min = -1.0
                lambda_max = 1.0
                lambda_t = (lambda_min + lambda_max) / 2.0
                
                model.eval()

                for step in range(100):
                    # 4) Update skill vector
                    IIP_z = z1 - lambda_t * z2  # this depends on λ

                    # 5) Begin rollout and collect history of (s, r)
                    reward_total = 0.0

                    observation, _ = env2.reset(seed=seed)
                    done = False
                    while(not done):
                        obs = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=IIP_z.device)
                        action = rew_model.act(obs, IIP_z).ravel()
                        observation, reward, terminated, truncated, info = env2.step(action)
                        reward_total += reward
                        done = bool(terminated or truncated)
                    print(reward_total)
                    # print(lambda_t)

                    # Set a threshold reward
                    threshold = 5.0
                    eps = 0.1

                    if (abs(reward_total - threshold) < eps):
                        break
                    elif (reward_total > threshold): # increase lambda
                        lambda_min = lambda_t
                    elif (reward_total < threshold): # decrease lambda
                        lambda_max = lambda_t
                    lambda_t = 0.5 * (lambda_min + lambda_max)

                video_dir = "videos"
                os.makedirs(video_dir, exist_ok=True)
                
                # 1. z1 rollout in env1 (Goal 1)
                frames_z1 = []
                obs, _ = env1.reset(seed=seed)
                done = False
                while not done:
                    obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=z1.device)
                    action = rew_model.act(obs_tensor, z1).ravel()
                    obs, _, terminated, truncated, _ = env1.step(action)
                    frame = env1.render()
                    frames_z1.append(frame)
                    done = bool(terminated or truncated)

                # 2. z2 rollout in env2 (Goal 2)
                frames_z2 = []
                obs, _ = env2.reset(seed=seed)
                done = False
                while not done:
                    obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=z2.device)
                    action = rew_model.act(obs_tensor, z2).ravel()
                    obs, _, terminated, truncated, _ = env2.step(action)
                    frame = env2.render()
                    frames_z2.append(frame)
                    done = bool(terminated or truncated)

                # 3. IIP rollout in env1
                frames_iip = []
                obs, _ = env1.reset(seed=seed)
                done = False
                while not done:
                    IIP_z = z1 - lambda_t * z2
                    obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=IIP_z.device)
                    action = rew_model.act(obs_tensor, IIP_z).ravel()
                    obs, _, terminated, truncated, _ = env1.step(action)
                    frame = env1.render()
                    frames_iip.append(frame)
                    done = bool(terminated or truncated)

                # Normalize all frame lists to same length by padding last frame
                max_len = max(len(frames_z1), len(frames_z2), len(frames_iip))
                def pad_frames(frames, max_len):
                    if len(frames) < max_len:
                        last = frames[-1]
                        frames += [last] * (max_len - len(frames))
                    return frames

                frames_z1 = pad_frames(frames_z1, max_len)
                frames_z2 = pad_frames(frames_z2, max_len)
                frames_iip = pad_frames(frames_iip, max_len)

                # Add labels
                frames_z1 = add_text_to_frames(frames_z1, "Goal 1")
                frames_z2 = add_text_to_frames(frames_z2, "Goal 2")
                frames_iip = add_text_to_frames(frames_iip, "IIP")

                # Combine frames horizontally
                combined_frames = [np.hstack((f1, f2, f3)) for f1, f2, f3 in zip(frames_z1, frames_z2, frames_iip)]

                # Save combined video
                output_path = os.path.join(video_dir, f"{task1}_to_{task2}_IIP_compare.mp4")
                media.write_video(output_path, combined_frames, fps=30)
                print(f"🎥 Saved comparison video to {output_path}")


            # reward_mean = np.mean(task_rewards)
            # reward_std = np.std(task_rewards)
            # result = (
            #     f"✅ Reward: {reward_mean:.2f} ± {reward_std:.2f}\n"
            # )
            # print(result)
            # f.write(result)

    print(f"\n📄 All results saved to {output_file}")
