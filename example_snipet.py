import os
os.environ["OMP_NUM_THREADS"] = "1"
from humenv import STANDARD_TASKS
from humenv import make_humenv
import mediapy as media
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper

from metamotivo.buffers.buffers import DictBuffer
from huggingface_hub import hf_hub_download
import h5py
import torch
import gymnasium

if __name__ == "__main__":
    local_dir = "metamotivo-S-1-datasets"
    dataset = "buffer_inference_500000.hdf5"  # a smaller buffer that can be used for reward inference
    # dataset = "buffer.hdf5"  # the full training buffer of the model
    buffer_path = hf_hub_download(
            repo_id="facebook/metamotivo-S-1",
            filename=f"data/{dataset}",
            repo_type="model",
            local_dir=local_dir,
        )
    hf = h5py.File(buffer_path, "r")
    # print(hf.keys())

    # create a DictBuffer object that can be used for sampling
    data = {k: v[:] for k, v in hf.items()}
    buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
    buffer.extend(data)

    task = STANDARD_TASKS[1]
    print(task)
    model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", device="cpu")
    rew_model = RewardWrapper(
            model=model,
            inference_dataset=buffer, # see above how to download and create a buffer
            num_samples_per_inference=100_000,
            inference_function="reward_wr_inference",
            max_workers=40,
            process_executor=True,
            process_context="forkserver"
        )
    z = rew_model.reward_inference(task)
    env, _ = make_humenv(num_envs=1, task=task, state_init="DefaultAndFall", wrappers=[gymnasium.wrappers.FlattenObservation])
    done = False
    observation, info = env.reset()
    frames = [env.render()]
    while not done:
        obs = torch.tensor(observation.reshape(1,-1), dtype=torch.float32, device=rew_model.device)
        action = rew_model.act(obs=obs, z=z).ravel()
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        done = bool(terminated or truncated)

    media.write_video(f"videos/metamotivo_task_{task}.mp4", frames, fps=30)
