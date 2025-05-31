import os
os.environ["OMP_NUM_THREADS"] = "1"

import gymnasium
import torch
import numpy as np
import h5py
import mediapy as media

from huggingface_hub import hf_hub_download
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper
from metamotivo.buffers.buffers import DictBuffer
from humenv import STANDARD_TASKS, make_humenv

if __name__ == "__main__":
    # Download dataset
    local_dir = "metamotivo-S-1-datasets"
    dataset = "buffer_inference_500000.hdf5"
    buffer_path = hf_hub_download(
        repo_id="facebook/metamotivo-S-1",
        filename=f"data/{dataset}",
        repo_type="model",
        local_dir=local_dir,
    )

    # Load into DictBuffer
    hf = h5py.File(buffer_path, "r")
    data = {k: v[:] for k, v in hf.items()}
    buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
    buffer.extend(data)

    # Setup model and task
    task = STANDARD_TASKS[1]
    print("Task:", task)
    device = "cpu"
    model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", device=device)

    rew_model = RewardWrapper(
        model=model,
        inference_dataset=buffer,
        num_samples_per_inference=100_000,
        inference_function="reward_wr_inference",
        max_workers=40,
        process_executor=True,
        process_context="forkserver",
    )

    z = rew_model.reward_inference(task)

    env, _ = make_humenv(
        num_envs=1,
        task=task,
        state_init="DefaultAndFall",
        wrappers=[gymnasium.wrappers.FlattenObservation]
    )
    observation, info = env.reset()
    done = False
    frames = [env.render()]

    # Cost constraint setup
    specific_dimensions = [-1]
    sample_range = [-1.0, 1.0]

    observation_c_z = z.clone()
    for dim in specific_dimensions:
        random_samples = np.random.uniform(sample_range[0], sample_range[1], size=observation_c_z.shape[0])
        observation_c_z[:, dim] = torch.tensor(random_samples, dtype=torch.float32).to(observation_c_z.device)

    reward_term = 10 * (torch.tensor(random_samples, dtype=torch.float32).to(observation_c_z.device).unsqueeze(1) - sample_range[0])
    c_z = model.reward_wr_inference(observation_c_z, reward_term)

    # Lagrange multiplier search
    lagrange_Min, lagrange_Max = 0, 10
    eta = -100
    lagrange_min = lagrange_Min
    lagrange_max = lagrange_Max
    lagrange_multiplier = np.random.uniform(lagrange_min, lagrange_max)

    obs_torch = torch.as_tensor(observation["observations"], dtype=torch.float32, device=z.device).unsqueeze(0)

    with torch.no_grad():
        while abs(lagrange_max - lagrange_min) > 1e-5:
            Z_lambda_c = z - lagrange_multiplier * c_z
            action = model.actor(obs_torch, Z_lambda_c, std=0.0)
            Q = model.critic(obs_torch, Z_lambda_c, action).squeeze()

            if abs(Q.item() - eta) < 1e-2:
                break
            elif Q.item() > eta:
                lagrange_min = lagrange_multiplier
            else:
                lagrange_max = lagrange_multiplier

            lagrange_multiplier = (lagrange_min + lagrange_max) / 2

    # Final rollout
    task_reward = 0.0
    task_cost = 0.0
    done = False
    observation, _ = env.reset()
    frames = [env.render()]

    while not done:
        Z_lambda_c = z - lagrange_multiplier * c_z
        obs = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=z.device)
        action = model.actor(obs, Z_lambda_c, std=0.0).ravel().numpy()
        observation, reward, terminated, truncated, info = env.step(action)

        task_reward += reward
        for dim in specific_dimensions:
            if observation[dim] > sample_range[0]:
                task_cost += 10 * (observation[dim] - sample_range[0])

        frames.append(env.render())
        done = bool(terminated or truncated)

    # Save video
    os.makedirs("videos", exist_ok=True)
    video_path = f"videos/metamotivo_task_{task}_lagrange.mp4"
    media.write_video(video_path, frames, fps=30)

    # Print results
    print(f"✅ Finished rollout for task: {task}")
    print(f"🎯 Total reward: {task_reward}")
    print(f"⚠️ Total cost: {task_cost}")
    print(f"📹 Video saved at: {video_path}")
