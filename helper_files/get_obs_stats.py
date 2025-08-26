import os
import numpy as np
import torch
import h5py
import gymnasium
import mediapy as media
from huggingface_hub import hf_hub_download

from humenv import STANDARD_TASKS, make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper
from metamotivo.buffers.buffers import DictBuffer

os.environ["OMP_NUM_THREADS"] = "1"

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

    os.makedirs("videos", exist_ok=True)
    stats_log_path = "metamotivo_observation_stats.txt"
    with open(stats_log_path, "w") as stats_file:
        for task in STANDARD_TASKS:
            print(f"Processing task: {task}")
            stats_file.write(f"Task: {task}\n")

            z = rew_model.reward_inference(task)
            env, _ = make_humenv(num_envs=1, task=task, state_init="DefaultAndFall",
                                 wrappers=[gymnasium.wrappers.FlattenObservation])

            done = False
            observation, info = env.reset()
            obs_log = [observation.copy()]
            frames = [env.render()]

            while not done:
                obs_tensor = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=rew_model.device)
                action = rew_model.act(obs=obs_tensor, z=z).ravel()
                observation, reward, terminated, truncated, info = env.step(action)
                obs_log.append(observation.copy())
                frames.append(env.render())
                done = bool(terminated or truncated)

            media.write_video(f"videos/metamotivo_task_{task}.mp4", frames, fps=30)

            # Convert obs_log to numpy array for statistics
            obs_array = np.array(obs_log)
            dim = obs_array.shape[1]

            for i in range(dim):
                dim_stats = {
                    "min": np.min(obs_array[:, i]),
                    "max": np.max(obs_array[:, i]),
                    "mean": np.mean(obs_array[:, i]),
                    "std": np.std(obs_array[:, i]),
                }
                stats_file.write(
                    f"  Dimension {i}: min={dim_stats['min']:.4f}, max={dim_stats['max']:.4f}, "
                    f"mean={dim_stats['mean']:.4f}, std={dim_stats['std']:.4f}\n"
                )
            stats_file.write("\n")
    print(f"All task videos saved in ./videos/, statistics written to {stats_log_path}")
