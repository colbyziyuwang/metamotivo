import os
import numpy as np
import torch
import h5py
import gymnasium
import random
from huggingface_hub import hf_hub_download
from humenv import STANDARD_TASKS, make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper
from metamotivo.buffers.buffers import DictBuffer
from utils import set_seed

os.environ["OMP_NUM_THREADS"] = "1"
METHOD = "bisection"
random.seed(42)

def evaluate_reward(env, rew_model, skill_z, seed):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=skill_z.device)
        action = rew_model.act(obs_tensor, skill_z).ravel()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = bool(terminated or truncated)
    return total_reward

if __name__ == "__main__":
    # Load dataset from Hugging Face
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

    # Load reward model
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

    output_file = f"auto_IIP_eval_per_seed_threshold_{METHOD}.txt"
    with open(output_file, "w") as f:
        for task1 in STANDARD_TASKS:
            task_pool = [t for t in STANDARD_TASKS if t != task1]
            random.seed(42)
            sampled_tasks = random.sample(task_pool, 3)

            for task2 in sampled_tasks:
                print(f"\n🎯 task1: {task1} | task2: {task2}")
                f.write(f"task1: {task1}\n")
                f.write(f"task2: {task2}\n")

                z1 = rew_model.reward_inference(task1)
                z2 = rew_model.reward_inference(task2)

                rewards_z1_env1 = []
                rewards_z1_env2 = []
                rewards_iip_env1 = []
                rewards_iip_env2 = []

                for seed in range(5):
                    set_seed(seed)

                    env1, _ = make_humenv(num_envs=1, task=task1, state_init="Default", seed=seed,
                                          wrappers=[gymnasium.wrappers.FlattenObservation])
                    env2, _ = make_humenv(num_envs=1, task=task2, state_init="Default", seed=seed,
                                          wrappers=[gymnasium.wrappers.FlattenObservation])

                    # Evaluate z1 in both envs
                    r1 = evaluate_reward(env1, rew_model, z1, seed)
                    r2 = evaluate_reward(env2, rew_model, z1, seed)
                    threshold = 0.5 * r2  # 🔁 per-seed threshold

                    print(f"[Seed {seed}] r1 = {r1:.2f}, r2 = {r2:.2f} → threshold = {threshold:.2f}")

                    # Bisection search for lambda
                    lambda_min, lambda_max = -1.0, 1.0
                    lambda_t = 0.5 * (lambda_min + lambda_max)

                    for step in range(100):
                        z_iip = z1 - lambda_t * z2
                        reward_iip = evaluate_reward(env2, rew_model, z_iip, seed)
                        if abs(reward_iip - threshold) < 0.1:
                            break
                        elif reward_iip > threshold:
                            lambda_min = lambda_t
                        else:
                            lambda_max = lambda_t
                        lambda_t = 0.5 * (lambda_min + lambda_max)

                    final_z_iip = z1 - lambda_t * z2
                    r3 = evaluate_reward(env1, rew_model, final_z_iip, seed)
                    r4 = evaluate_reward(env2, rew_model, final_z_iip, seed)

                    rewards_z1_env1.append(r1)
                    rewards_z1_env2.append(r2)
                    rewards_iip_env1.append(r3)
                    rewards_iip_env2.append(r4)

                # Write results per task pair
                def write_metric(name, values):
                    mean = np.mean(values)
                    std = np.std(values)
                    f.write(f"{name}: {mean:.2f} ± {std:.2f}\n")

                write_metric("reward_z1_env1", rewards_z1_env1)
                write_metric("reward_z1_env2", rewards_z1_env2)
                write_metric("reward_iip_env1", rewards_iip_env1)
                write_metric("reward_iip_env2", rewards_iip_env2)
                f.write("\n")

    print(f"\n✅ All results saved to: {output_file}")
