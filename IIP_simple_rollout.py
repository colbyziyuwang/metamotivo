import os
import numpy as np
import torch
import h5py
import gymnasium
import random
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from humenv import STANDARD_TASKS, make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper
from metamotivo.buffers.buffers import DictBuffer
from utils import set_seed

os.environ["OMP_NUM_THREADS"] = "1"
METHOD = "bisection"  # bisection, baseline (step-based)
random.seed(42)

# Setup plot directory
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

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

def save_boxplot(data_dict, title, ylabel, filename):
    labels = list(data_dict.keys())
    data = list(data_dict.values())

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

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

    output_file = f"auto_IIP_eval_per_seed_threshold_{METHOD}.txt"
    with open(output_file, "w") as f:
        # For global collection across all task pairs
        all_changes_env1 = []
        all_changes_env2 = []

        for task1 in STANDARD_TASKS:
            task_pool = [t for t in STANDARD_TASKS if t != task1]
            sampled_tasks = random.sample(task_pool, 3)

            for task2 in sampled_tasks:
                print(f"\n🎯 task1: {task1} | task2: {task2}")
                f.write(f"task1: {task1}\n")
                f.write(f"task2: {task2}\n")

                rewards_z1_env1 = []
                rewards_z1_env2 = []
                rewards_iip_env1 = []
                rewards_iip_env2 = []

                for seed in range(5):
                    set_seed(seed)

                    z1 = rew_model.reward_inference(task1)
                    z2 = rew_model.reward_inference(task2)

                    env1, _ = make_humenv(num_envs=1, task=task1, state_init="Default", seed=seed,
                                          wrappers=[gymnasium.wrappers.FlattenObservation])
                    env2, _ = make_humenv(num_envs=1, task=task2, state_init="Default", seed=seed,
                                          wrappers=[gymnasium.wrappers.FlattenObservation])

                    r1 = evaluate_reward(env1, rew_model, z1, seed)
                    r2 = evaluate_reward(env2, rew_model, z1, seed)
                    threshold = 0.5 * r2
                    print(f"[Seed {seed}] r1 = {r1:.2f}, r2 = {r2:.2f} → threshold = {threshold:.2f}")

                    lambda_min, lambda_max = -1.0, 1.0
                    lambda_t = 0.5 * (lambda_min + lambda_max)
                    step_size = 0.01

                    for step in range(100):
                        z_iip = z1 - lambda_t * z2
                        reward_iip = evaluate_reward(env2, rew_model, z_iip, seed)

                        if METHOD == "bisection":
                            if abs(reward_iip - threshold) < 0.1:
                                break
                            elif reward_iip > threshold:
                                lambda_min = lambda_t
                            else:
                                lambda_max = lambda_t
                            lambda_t = 0.5 * (lambda_min + lambda_max)
                        elif METHOD == "baseline":
                            if abs(reward_iip - threshold) < 0.1:
                                break
                            elif reward_iip > threshold:
                                lambda_t += step_size
                            else:
                                lambda_t -= step_size
                            lambda_t = max(min(lambda_t, lambda_max), lambda_min)

                    final_z_iip = z1 - lambda_t * z2
                    r3 = evaluate_reward(env1, rew_model, final_z_iip, seed)
                    r4 = evaluate_reward(env2, rew_model, final_z_iip, seed)

                    rewards_z1_env1.append(r1)
                    rewards_z1_env2.append(r2)
                    rewards_iip_env1.append(r3)
                    rewards_iip_env2.append(r4)

                # Compute percentage changes
                pc_env1 = 100.0 * (np.array(rewards_iip_env1) - np.array(rewards_z1_env1)) / np.maximum(np.abs(rewards_z1_env1), 1e-8)
                pc_env2 = 100.0 * (np.array(rewards_iip_env2) - np.array(rewards_z1_env2)) / np.maximum(np.abs(rewards_z1_env2), 1e-8)

                all_changes_env1.extend(pc_env1.tolist())
                all_changes_env2.extend(pc_env2.tolist())

                # Print mean ± std instead of boxplots
                def mean_std(arr):
                    return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"

                result_lines = [
                    f"reward_z1_env1: {mean_std(rewards_z1_env1)}",
                    f"reward_z1_env2: {mean_std(rewards_z1_env2)}",
                    f"reward_iip_env1: {mean_std(rewards_iip_env1)}",
                    f"reward_iip_env2: {mean_std(rewards_iip_env2)}"
                ]
                for line in result_lines:
                    print(line)
                    f.write(line + "\n")
                f.write("\n")

        # Save global summary boxplots
        save_boxplot({
            "Env1 Δ% (All Pairs)": all_changes_env1,
            "Env2 Δ% (All Pairs)": all_changes_env2
        }, title="All Task Pairs % Reward Change", ylabel="Δ% Reward", filename=f"global_delta_pct_{METHOD}.png")

    print(f"\n✅ All results saved to: {output_file}, global boxplots saved to {plot_dir}")
