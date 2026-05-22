from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm



def root_local_rot_tan_norm(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)

    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)

    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local) # shape: (num_envs, num_steps, 3, 3)

    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs


def amp_body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Key body positions in the robot root frame, matching mjlab's body_pos_b style."""

    robot: Articulation = env.scene[asset_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    num_bodies = body_pos_w.shape[1]
    body_pos_b = math_utils.quat_apply_inverse(
        root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1),
        body_pos_w - root_pos_w.unsqueeze(1),
    )
    return body_pos_b.reshape(env.num_envs, -1)


def amp_body_lin_vel_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Key body linear velocities in the robot root frame.

    The stored pkl motions expose key-body positions but not per-body orientations.
    Root-frame relative velocities are therefore the closest shared representation
    available for both policy and demonstration AMP observations.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    root_lin_vel_w = robot.data.root_lin_vel_w
    root_quat_w = robot.data.root_quat_w
    num_bodies = body_lin_vel_w.shape[1]
    body_lin_vel_b = math_utils.quat_apply_inverse(
        root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1),
        body_lin_vel_w - root_lin_vel_w.unsqueeze(1),
    )
    return body_lin_vel_b.reshape(env.num_envs, -1)


def ref_amp_body_pos_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    """Reference key body positions in the root frame."""

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_body_pos_b = animation_term.get_key_body_pos_b()
    if flatten_steps_dim:
        return ref_body_pos_b.reshape(env.num_envs, -1)
    return ref_body_pos_b.reshape(env.num_envs, ref_body_pos_b.shape[1], -1)


def ref_amp_body_lin_vel_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    """Reference key body relative linear velocities from finite differences."""

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_body_pos_b = animation_term.get_key_body_pos_b()
    ref_body_lin_vel_b = torch.zeros_like(ref_body_pos_b)
    if ref_body_pos_b.shape[1] > 1:
        ref_body_lin_vel_b[:, :-1] = (ref_body_pos_b[:, 1:] - ref_body_pos_b[:, :-1]) / env.step_dt
        ref_body_lin_vel_b[:, -1] = ref_body_lin_vel_b[:, -2]
    if flatten_steps_dim:
        return ref_body_lin_vel_b.reshape(env.num_envs, -1)
    return ref_body_lin_vel_b.reshape(env.num_envs, ref_body_lin_vel_b.shape[1], -1)
