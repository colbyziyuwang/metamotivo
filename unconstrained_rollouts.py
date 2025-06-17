import os
import numpy as np
import torch
import h5py
import gymnasium
from huggingface_hub import hf_hub_download

from humenv import STANDARD_TASKS, make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper
from metamotivo.buffers.buffers import DictBuffer
from set_seed import set_seed
import mediapy as media

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

    sample_range = (-1.0, 1.0)
    specific_dimensions = [-1]
    output_file = "unconstrained_output.txt"

    with open(output_file, "w") as f:
        for task in STANDARD_TASKS:
            print(f"\n🎯 Task: {task}")
            f.write(f"\n🎯 Task: {task}\n")
            z = rew_model.reward_inference(task)

            task_rewards = []
            task_costs = []

            for seed in range(5):
                set_seed(seed)
                env, _ = make_humenv(num_envs=1, task=task, state_init="DefaultAndFall", seed=seed,
                                     wrappers=[gymnasium.wrappers.FlattenObservation])
                observation, info = env.reset(seed=seed)
                done = False

                total_reward = 0.0
                task_cost = 0.0

                frames = [env.render()]
                while not done:
                    obs_tensor = torch.tensor(observation.reshape(1, -1), dtype=torch.float32, device=rew_model.device)
                    action = rew_model.act(obs=obs_tensor, z=z, mean=True).ravel()
                    observation, reward, terminated, truncated, info = env.step(action)
                    frames.append(env.render())
                    done = bool(terminated or truncated)
                    total_reward += reward

                    for dim in specific_dimensions:
                        if observation[dim] > sample_range[0]:
                            task_cost += 10 * (observation[dim] - sample_range[0])

                task_rewards.append(total_reward)
                task_costs.append(task_cost)

            reward_mean = np.mean(task_rewards)
            reward_std = np.std(task_rewards)
            cost_mean = np.mean(task_costs)
            cost_std = np.std(task_costs)

            result = (
                f"✅ Reward: {reward_mean:.2f} ± {reward_std:.2f}\n"
                f"⚠️ Cost:   {cost_mean:.2f} ± {cost_std:.2f}\n"
            )
            print(result)
            f.write(result)

            # save videos
            video_dir = "videos"
            os.makedirs(video_dir, exist_ok=True)
            range_str = f"range{sample_range[0]}to{sample_range[1]}"
            dims_str = f"dims{'_'.join(map(str, specific_dimensions))}"
            video_filename = f"{task.replace('/', '_')}_seed{seed}_{range_str}_{dims_str}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            media.write_video(video_path, frames, fps=30)

    print(f"\n📄 All results saved to {output_file}")
