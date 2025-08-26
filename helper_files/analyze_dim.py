import sys
import inspect
import numpy as np
import mujoco
import gymnasium as gym
from enum import Enum
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from pathlib import Path
import humenv.reset
import humenv.rewards
import humenv.utils
from humenv.misc.motionlib import MotionBuffer
from humenv import make_humenv


_XML = "assets/robot.xml"  # this is a copy of robot_july5_mpd_kp3_kd2.xml
_ROBOT_IDX_START: int = 1
_ROBOT_IDX_END: int = 25
_NUM_RIGID_BODIES: int = 24
_NUM_VEL_LIMIT: int = 72

def compute_humanoid_self_obs_v2(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    upright_start: bool,
    root_height_obs: bool,
    humanoid_type: str,
) -> Dict[str, np.ndarray]:
    body_pos = data.xpos.copy()[_ROBOT_IDX_START:_ROBOT_IDX_END][None,]
    body_rot = data.xquat.copy()[_ROBOT_IDX_START:_ROBOT_IDX_END][None,]
    body_vel = data.sensordata[:_NUM_VEL_LIMIT].reshape(_NUM_RIGID_BODIES, 3).copy()[None,]
    body_ang_vel = data.sensordata[_NUM_VEL_LIMIT : 2 * _NUM_VEL_LIMIT].reshape(_NUM_RIGID_BODIES, 3).copy()[None,]

    obs = OrderedDict()

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    if not upright_start:
        root_rot = humenv.utils.remove_base_rot(root_rot, humanoid_type)

    heading_rot_inv = humenv.utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:, 2:3]

    if root_height_obs:
        obs["root_h_obs"] = root_h

    heading_rot_inv_expand = heading_rot_inv[..., None, :]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1], axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
        heading_rot_inv_expand.shape[2],
    )

    root_pos_expand = root_pos[..., None, :]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = humenv.utils.quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    obs["local_body_pos"] = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = humenv.utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = humenv.utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    ###### Velocity ######
    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = humenv.utils.quat_rotate(flat_heading_rot_inv, flat_body_vel)
    obs["local_body_vel"] = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = humenv.utils.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    return obs

env, _ = make_humenv()
env.reset()

# access raw mujoco model and data
model = env.unwrapped.model
data = env.unwrapped.data

# ensure kinematics are updated
mujoco.mj_kinematics(model, data)

# compute observation dict
obs_dict = compute_humanoid_self_obs_v2(
    model,
    data,
    upright_start=False,
    root_height_obs=True,
    humanoid_type="smpl"
)

# print key shapes
for k, v in obs_dict.items():
    print(f"{k}: {v.shape}")
