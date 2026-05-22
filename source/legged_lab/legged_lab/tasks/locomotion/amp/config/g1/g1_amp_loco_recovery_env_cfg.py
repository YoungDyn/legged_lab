"""G1 AMP 行走与自恢复(跌倒起身) 联合训练任务配置。

该配置借鉴了 AMP_mjlab 的 locomotion+recovery 模式，并结合了 Legged Lab 的 pkl 动捕数据流，核心设计如下：
1. 双源运动数据：正常重置的环境从“走跑(walk_and_run)”数据中采样初始化；进入延迟重置的环境从“起身(get_up)”数据中采样。
2. 延迟终止(Delayed Termination)：抽取一部分比例的环境，在摔倒时屏蔽死亡判定，赋予其一个短时间的“恢复窗口”(Recovery Window)，强制模型在此期间学习起身。
3. 动态掩码(Masking)：在恢复窗口内，屏蔽追踪速度指令的奖励，仅激活“起身高度、姿态恢复”的专属奖励。
4. 惩罚隔离(防止躺平陷阱)：将正常行走时的能量、平滑度、加速度等惩罚在恢复期全部屏蔽，防止与起身所需的剧烈爆发动作产生奖励冲突。
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationManager
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.managers import AnimationTermCfg as AnimTerm
from legged_lab.managers import MotionDataTermCfg as MotionDataTerm

from .g1_amp_env_cfg import AMP_NUM_STEPS, KEY_BODY_NAMES, G1AmpEnvCfg as G1WalkRunAmpEnvCfg

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAmpEnv


LOCOMOTION_ANIMATION_TERM_NAME = "animation"
RECOVERY_ANIMATION_TERM_NAME = "recovery_animation"
RECOVERY_STANDUP_HEIGHT_THRESHOLD = 0.65
RECOVERY_ORIENTATION_HEIGHT_THRESHOLD = 0.55
RECOVERY_EXIT_HEIGHT_THRESHOLD = 0.68
RECOVERY_EXIT_TILT_ERROR_THRESHOLD = 0.25
RECOVERY_ASSIST_FORCE_N = 80.0  # 降低托举辅助力 (120N -> 80N)，减少直升机式的空翻。
RECOVERY_ASSIST_FADE_END_HEIGHT = 0.75

# 这里保留两套 motion data / animation term：
# - locomotion: 正常走跑数据，用于常规 reset、AMP demo、速度跟踪训练；
# - recovery: 摔倒后起身数据，用于延迟终止环境的 reset 和部分 AMP demo。
LOCOMOTION_MOTION_DATA_TERM_NAME = "motion_dataset"
RECOVERY_MOTION_DATA_TERM_NAME = "recovery_dataset"


def _walk_run_motion_weights() -> dict[str, float]:
    """Walk/run motion clips used by normal locomotion environments."""

    return {
        "B10_-__Walk_turn_left_45_stageii": 1.0,
        "B11_-__Walk_turn_left_135_stageii": 1.0,
        "B13_-__Walk_turn_right_90_stageii": 1.0,
        "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
        "B15_-__Walk_turn_around_stageii": 1.0,
        "B22_-__side_step_left_stageii": 1.0,
        "B23_-__side_step_right_stageii": 1.0,
        "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
        "B9_-__Walk_turn_left_90_stageii": 1.0,
        "C11_-_run_turn_left_90_stageii": 0.0,
        "C12_-_run_turn_left_45_stageii": 0.0,
        "C13_-_run_turn_left_135_stageii": 0.0,
        "C14_-_run_turn_right_90_stageii": 0.0,
        "C15_-_run_turn_right_45_stageii": 0.0,
        "C16_-_run_turn_right_135_stageii": 0.0,
        "C17_-_run_change_direction_stageii": 0.0,
        "C1_-_stand_to_run_stageii": 0.0,
        "C3_-_run_stageii": 0.0,
        "C4_-_run_to_walk_a_stageii": 0.0,
        "C5_-_walk_to_run_stageii": 0.0,
        "C6_-_stand_to_run_backwards_stageii": 0.0,
        "C8_-_run_backwards_to_stand_stageii": 0.0,
        "C9_-_run_backwards_turn_run_forward_stageii": 0.0,
        "Walk_B10_-_Walk_turn_left_45_stageii": 0.0,
        "Walk_B13_-_Walk_turn_right_45_stageii": 0.0,
        "Walk_B15_-_Walk_turn_around_stageii": 0.0,
        "Walk_B16_-_Walk_turn_change_stageii": 0.0,
        "Walk_B22_-_Side_step_left_stageii": 0.0,
        "Walk_B23_-_Side_step_right_stageii": 0.0,
        "Walk_B4_-_Stand_to_Walk_Back_stageii": 0.0,
    }


def _recovery_motion_weights() -> dict[str, float]:
    """Fall-and-get-up motion clips used by recovery environments."""

    return {
        "fallAndGetUp1_subject1_680_800": 0.0,
        "fallAndGetUp1_subject1_850_940": 0.0,
        "fallAndGetUp1_subject1_1060_1150": 0.0,
        "fallAndGetUp1_subject1_1400_1480": 0.0,
        "fallAndGetUp1_subject1_1600_1700": 0.0,
        "fallAndGetUp1_subject1_2100_2200": 0.0,
        "fallAndGetUp1_subject1_2300_2400": 0.0,
        "fallAndGetUp1_subject4_3700_3800": 0.0,
        "fallAndGetUp1_subject5_2100_2200": 0.0,
        "fallAndGetUp1_subject5_2500_2600": 0.0,
        "fallAndGetUp1_subject5_3900_3980": 0.0,
        "fallAndGetUp2_subject2_360_580": 0.0,
        "fallAndGetUp2_subject2_850_1050": 0.0,
        "fallAndGetUp2_subject2_1200_1370": 1.0,
        "fallAndGetUp2_subject2_1500_1600": 0.0,
        "fallAndGetUp2_subject3_450_550": 0.0,
        "fallAndGetUp2_subject3_900_1000": 0.0,
        "fallAndGetUp2_subject3_1850_1920": 0.0,
        "fallAndGetUp2_subject3_2080_2180": 0.0,
        "fallAndGetUp6_subject1_530_600": 0.0,
        "fallAndGetUp6_subject1_650_700": 0.0,
        "fallAndGetUp6_subject1_1080_1180": 0.0,
        "fallAndGetUp6_subject1_1230_1300": 0.0,
        "fallAndGetUp6_subject1_1630_1690": 0.0,
    }


def _manager_terminated_tensor(manager: TerminationManager) -> torch.Tensor:
    """Read the public termination buffer from Isaac Lab's manager."""

    return manager.terminated


def _set_manager_terminated(manager: TerminationManager, mask: torch.Tensor, value: bool) -> None:
    """Mutate Isaac Lab termination buffers across minor API naming changes."""

    try:
        manager.terminated[mask] = value
        return
    except Exception:
        pass

    for name in ("_terminated_buf", "_terminated", "terminated_buf"):
        buf = getattr(manager, name, None)
        if isinstance(buf, torch.Tensor):
            buf[mask] = value
            return

    raise AttributeError("Could not find mutable terminated buffer on TerminationManager.")


class DelayedRecoveryTerminationManager(TerminationManager):
    """
    延迟终止管理器 (Delayed Recovery Termination Manager)

    【核心状态机】：
    正常情况下，触发 bad_base_height 时环境会立即终止并重置。
    该类拦截了 termination 信号，对于被选中作为恢复训练的环境（delay_env_mask），
    给予一个最大 max_delay_steps（如 250 步）的“免死金牌”。在这个窗口期内，
    环境继续运行（赋予网络尝试起身的绝佳机会）。超时后若仍未恢复正常，则抛出真正的重置信号。
    """

    def __init__(
        self,
        base: TerminationManager,
        env: ManagerBasedAmpEnv,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
    ) -> None:
        # 复用已有 TerminationManager 的内部状态，只额外挂载恢复延迟需要的 mask/counter。
        self.__dict__.update(base.__dict__)
        self._recovery_env = env
        self._recovery_delay_env_mask = delay_env_mask
        self._recovery_delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._recovery_max_delay_steps = max_delay_steps

    def _recovery_success_mask(self) -> torch.Tensor:
        """Recovery exits only after the torso is high and roughly upright."""

        env = self._recovery_env
        robot: Articulation = env.scene["robot"]
        height_ok = robot.data.root_link_pos_w[:, 2] > RECOVERY_EXIT_HEIGHT_THRESHOLD
        tilt_error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1)
        upright_ok = tilt_error < RECOVERY_EXIT_TILT_ERROR_THRESHOLD
        return height_ok & upright_ok

    def compute(self) -> torch.Tensor:
        done = super().compute()

        if self._recovery_max_delay_steps <= 0:
            return done

        terminated = _manager_terminated_tensor(self)

        # Recovery is latched after the first fall termination.  Do not leave the
        # recovery reward window merely because the robot rose above the death
        # height; keep it active until the torso is high and upright.
        just_triggered = self._recovery_delay_env_mask & terminated
        active = self._recovery_delay_env_mask & ((self._recovery_delay_counters > 0) | just_triggered)
        self._recovery_delay_counters[active] += 1

        recovered = active & self._recovery_success_mask()
        self._recovery_delay_counters[recovered] = 0
        active = active & ~recovered

        not_ready = active & (self._recovery_delay_counters < self._recovery_max_delay_steps)
        if torch.any(not_ready):
            _set_manager_terminated(self, not_ready, False)

        # 达到最大延迟步数后退出恢复窗口；如果此时仍触发 termination，就允许正常 reset。
        expired = active & (self._recovery_delay_counters >= self._recovery_max_delay_steps)
        self._recovery_delay_counters[expired] = 0

        return torch.logical_or(self.terminated, self.time_outs)


def install_delayed_recovery_termination(
    env: ManagerBasedAmpEnv,
    env_ids: torch.Tensor | None,
    delay_env_ratio: float,
    max_delay_steps: int,
) -> None:
    """Startup event: install delayed termination for a random subset of envs."""

    if isinstance(env.termination_manager, DelayedRecoveryTerminationManager):
        return

    num_delay_envs = int(env.num_envs * delay_env_ratio)
    if num_delay_envs <= 0 or max_delay_steps <= 0:
        return

    # 随机抽一部分并行环境作为 recovery env；其他环境继续做普通 locomotion。
    delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    delay_ids = torch.randperm(env.num_envs, device=env.device)[:num_delay_envs]
    delay_mask[delay_ids] = True
    env.termination_manager = DelayedRecoveryTerminationManager(
        base=env.termination_manager,
        env=env,
        delay_env_mask=delay_mask,
        max_delay_steps=max_delay_steps,
    )
    print(
        "[loco_recovery] DelayedRecoveryTerminationManager installed: "
        f"{num_delay_envs}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
    )


def apply_recovery_upward_assist(
    env: ManagerBasedAmpEnv,
    env_ids: torch.Tensor | None,
    force_magnitude: float,
    fade_start_height: float,
    fade_end_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> None:
    """
    向上的恢复辅助力 (Upward Assist Force)
    机制：只在 active recovery (恢复窗口期) 内，从外部给机器人的躯干(torso)强行施加一个朝上的力（如120N）。
    目的：如同上帝之手轻轻提着机器人的背囊，极大地降低早期探索起身的难度，让动作噪声能快速触碰到成功的高度。
    衰减(fade)逻辑：当机器人躯干高度越过 0.65 并接近站立高度(0.75m)时，力度线性衰减到0，最终迫使机器人自己站稳。
    """

    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else robot.num_bodies
    forces = torch.zeros((len(env_ids), num_bodies, 3), device=env.device)
    torques = torch.zeros_like(forces)

    active = _active_recovery_mask(env)[env_ids]
    if torch.any(active):
        height = robot.data.root_link_pos_w[env_ids, 2]
        fade_span = max(fade_end_height - fade_start_height, 1.0e-6)
        scale = ((fade_end_height - height) / fade_span).clamp(0.0, 1.0)
        force_w = torch.zeros((len(env_ids), 3), device=env.device)
        force_w[:, 2] = torch.where(active, force_magnitude * scale, torch.zeros_like(scale))
        body_quat_w = robot.data.body_link_quat_w[env_ids][:, asset_cfg.body_ids[0]]
        forces[:, 0, :] = math_utils.quat_apply_inverse(body_quat_w, force_w)

    robot.set_external_force_and_torque(
        forces,
        torques,
        env_ids=env_ids,
        body_ids=asset_cfg.body_ids,
    )


def _active_recovery_mask(env: ManagerBasedAmpEnv) -> torch.Tensor:
    """Mask envs that are currently inside the delayed recovery window."""

    manager = env.termination_manager
    delay_mask = getattr(manager, "_recovery_delay_env_mask", None)
    counters = getattr(manager, "_recovery_delay_counters", None)
    if isinstance(delay_mask, torch.Tensor) and isinstance(counters, torch.Tensor):
        return delay_mask & (counters > 0)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _weak_recovery_reward(env: ManagerBasedAmpEnv, reward: torch.Tensor, weak_factor: float = 0.05) -> torch.Tensor:
    """
    【弱化掩码】：
    对于 active recovery 窗口内，不再简单粗暴地将惩罚置为 0，而是乘上一个极小的弱化系数（例如 0.05）。
    这样既保证了机器人有足够的力量起身，又微弱地引导它避免无意义的抽搐和空耗。
    """
    return torch.where(_active_recovery_mask(env), reward * weak_factor, reward)


def _normal_only_reward(env: ManagerBasedAmpEnv, reward: torch.Tensor) -> torch.Tensor:
    """
    【核心机制：奖励/惩罚隔离掩码 (Masking)】
    如果是 normal 正常行走状态，保留原有的奖励或惩罚。
    如果环境目前正处于 active recovery (摔倒在地，正在尝试起身的倒计时窗口中)，
    将这项奖励/惩罚直接强制置为 0。
    这一设计是破除“惩罚过高导致乱动分极低，从而直接躺平等死”这一局部最优陷阱的根本解决办法。
    """

    return torch.where(_active_recovery_mask(env), torch.zeros_like(reward), reward)


def _delay_env_mask(env: ManagerBasedAmpEnv) -> torch.Tensor | None:
    """Return the fixed recovery-env subset mask if delayed recovery is installed."""

    mask = getattr(env.termination_manager, "_recovery_delay_env_mask", None)
    return mask if isinstance(mask, torch.Tensor) else None


def _sample_motion_state(env: ManagerBasedAmpEnv, term_name: str, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Sample one frame from the requested motion dataset for each env."""

    motion_term = env.motion_data_manager.get_term(term_name)
    motion_ids = motion_term.sample_motions(len(env_ids))
    motion_times = motion_term.sample_times(motion_ids, truncate_time_end=env.step_dt)
    return motion_term.get_motion_state(motion_ids, motion_times)


def _write_motion_state_to_robot(
    env: ManagerBasedAmpEnv,
    env_ids: torch.Tensor,
    motion_state: dict[str, torch.Tensor],
    asset_cfg: SceneEntityCfg,
    height_offset: float,
) -> None:
    """Write sampled root/joint state into the robot simulation buffers."""

    robot: Articulation = env.scene[asset_cfg.name]

    # root_pos_w in motion data is relative to the mocap world; shift it to each env origin.
    root_pos = env.scene.env_origins[env_ids].clone()
    root_pos[:, 2] += motion_state["root_pos_w"][:, 2] + height_offset
    root_pose = torch.cat([root_pos, motion_state["root_quat"]], dim=-1)
    root_vel = torch.cat([motion_state["root_vel_w"], motion_state["root_ang_vel_w"]], dim=-1)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    joint_pos = motion_state["dof_pos"]
    joint_vel = motion_state["dof_vel"]

    # Motion data may slightly exceed the robot's configured soft limits after retargeting.
    # Clamp before writing to sim to avoid invalid reset states.
    if robot.data.soft_joint_pos_limits is not None:
        limits = robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    if robot.data.soft_joint_vel_limits is not None:
        vel_limits = robot.data.soft_joint_vel_limits[env_ids]
        joint_vel = joint_vel.clamp(-vel_limits, vel_limits)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_from_locomotion_or_recovery(
    env: ManagerBasedAmpEnv,
    env_ids: torch.Tensor,
    locomotion_motion_data_term: str,
    recovery_motion_data_term: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.05,
) -> None:
    """Reset normal envs from walk/run and delayed envs from recovery pkl data."""

    if env_ids is None or len(env_ids) == 0:
        return

    # 根据 startup 时固定下来的 delay_mask，把本次 reset 的 env 拆成两组：
    # 普通环境从 walk/run 采样，恢复环境从 get_up 采样。
    delay_mask = _delay_env_mask(env)
    if delay_mask is None:
        locomotion_ids = env_ids
        recovery_ids = env_ids[:0]
    else:
        is_delay_env = delay_mask[env_ids]
        locomotion_ids = env_ids[~is_delay_env]
        recovery_ids = env_ids[is_delay_env]

    if len(locomotion_ids) > 0:
        state = _sample_motion_state(env, locomotion_motion_data_term, locomotion_ids)
        _write_motion_state_to_robot(env, locomotion_ids, state, asset_cfg, height_offset)

    if len(recovery_ids) > 0:
        state = _sample_motion_state(env, recovery_motion_data_term, recovery_ids)
        _write_motion_state_to_robot(env, recovery_ids, state, asset_cfg, height_offset)


def track_lin_vel_xy_exp_normal_only(env, command_name: str, std: float) -> torch.Tensor:
    """Track commanded xy velocity except during the active recovery window."""

    reward = mdp.track_lin_vel_xy_exp(env, command_name=command_name, std=std)
    return _normal_only_reward(env, reward)


def track_ang_vel_z_exp_normal_only(env, command_name: str, std: float) -> torch.Tensor:
    """Track commanded yaw velocity except during the active recovery window."""

    reward = mdp.track_ang_vel_z_exp(env, command_name=command_name, std=std)
    return _normal_only_reward(env, reward)


def lin_vel_z_l2_normal_only(
    env: ManagerBasedAmpEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical base velocity only during normal locomotion."""

    return _weak_recovery_reward(env, mdp.lin_vel_z_l2(env, asset_cfg=asset_cfg), 0.05)


def ang_vel_xy_l2_normal_only(
    env: ManagerBasedAmpEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity only during normal locomotion."""

    return _weak_recovery_reward(env, mdp.ang_vel_xy_l2(env, asset_cfg=asset_cfg), 0.01)


def joint_torques_l2_normal_only(
    env: ManagerBasedAmpEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint effort only during normal locomotion."""

    return _weak_recovery_reward(env, mdp.joint_torques_l2(env, asset_cfg=asset_cfg), 0.01)


def joint_acc_l2_normal_only(
    env: ManagerBasedAmpEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint acceleration only during normal locomotion."""

    return _weak_recovery_reward(env, mdp.joint_acc_l2(env, asset_cfg=asset_cfg), 0.01)


def action_rate_l2_normal_only(env: ManagerBasedAmpEnv) -> torch.Tensor:
    """Penalize action changes only during normal locomotion."""

    return _weak_recovery_reward(env, mdp.action_rate_l2(env), 0.01)


def feet_slide_normal_only(
    env: ManagerBasedAmpEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot sliding only during normal locomotion."""

    return _normal_only_reward(env, mdp.feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg))


def feet_air_time_positive_biped_normal_only(
    env: ManagerBasedAmpEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward stepping only during normal locomotion."""

    reward = mdp.feet_air_time_positive_biped(env, command_name=command_name, threshold=threshold, sensor_cfg=sensor_cfg)
    return _normal_only_reward(env, reward)


def recovery_target_base_height(
    env: ManagerBasedAmpEnv,
    base_height_target: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward recovery envs for bringing the torso/root height back near standing height."""

    robot: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(robot.data.root_link_pos_w[:, 2] - base_height_target)
    reward = torch.exp(-height_error / (std * std))
    return torch.where(_active_recovery_mask(env), reward, torch.zeros_like(reward))


def recovery_target_orientation(
    env: ManagerBasedAmpEnv,
    std: float,
    standup_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward recovery envs for returning upright after the torso is high enough."""

    robot: Articulation = env.scene[asset_cfg.name]
    tilt_error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1)
    reward = torch.exp(-tilt_error / (std * std))
    mask = _active_recovery_mask(env) & (robot.data.root_link_pos_w[:, 2] > standup_height_threshold)
    return torch.where(mask, reward, torch.zeros_like(reward))


def recovery_low_body_velocity(
    env: ManagerBasedAmpEnv,
    lin_std: float,
    ang_std: float,
    standup_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward recovery envs for settling only after the torso is high enough."""

    robot: Articulation = env.scene[asset_cfg.name]
    lin_error = torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2]), dim=-1)
    ang_error = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=-1)
    reward = torch.exp(-lin_error / (lin_std * lin_std)) * torch.exp(-ang_error / (ang_std * ang_std))
    mask = _active_recovery_mask(env) & (robot.data.root_link_pos_w[:, 2] > standup_height_threshold)
    return torch.where(mask, reward, torch.zeros_like(reward))


def _demo_mask(env: ManagerBasedAmpEnv, recovery_demo_ratio: float) -> torch.Tensor:
    """Sample and cache per-step AMP demo selection between locomotion and recovery data."""

    step = int(getattr(env, "common_step_counter", 0))
    cached_step = getattr(env, "_loco_recovery_demo_mask_step", None)
    cached_mask = getattr(env, "_loco_recovery_demo_mask", None)
    if cached_step != step or not isinstance(cached_mask, torch.Tensor):
        cached_mask = torch.rand(env.num_envs, device=env.device) < recovery_demo_ratio
        env._loco_recovery_demo_mask = cached_mask
        env._loco_recovery_demo_mask_step = step
    return cached_mask


def _mix_demo_tensor(
    env: ManagerBasedAmpEnv,
    locomotion_value: torch.Tensor,
    recovery_value: torch.Tensor,
    recovery_demo_ratio: float,
) -> torch.Tensor:
    """Blend locomotion/recovery demo observations with the cached per-env mask."""

    mask = _demo_mask(env, recovery_demo_ratio)
    view_shape = (env.num_envs,) + (1,) * (locomotion_value.ndim - 1)
    return torch.where(mask.view(view_shape), recovery_value, locomotion_value)


def mixed_ref_root_local_rot_tan_norm(
    env: ManagerBasedAmpEnv,
    locomotion_animation: str,
    recovery_animation: str,
    recovery_demo_ratio: float,
    flatten_steps_dim: bool = False,
) -> torch.Tensor:
    locomotion = mdp.ref_root_local_rot_tan_norm(env, locomotion_animation, flatten_steps_dim)
    recovery = mdp.ref_root_local_rot_tan_norm(env, recovery_animation, flatten_steps_dim)
    return _mix_demo_tensor(env, locomotion, recovery, recovery_demo_ratio)


def mixed_ref_root_ang_vel_b(
    env: ManagerBasedAmpEnv,
    locomotion_animation: str,
    recovery_animation: str,
    recovery_demo_ratio: float,
    flatten_steps_dim: bool = False,
) -> torch.Tensor:
    locomotion = mdp.ref_root_ang_vel_b(env, locomotion_animation, flatten_steps_dim)
    recovery = mdp.ref_root_ang_vel_b(env, recovery_animation, flatten_steps_dim)
    return _mix_demo_tensor(env, locomotion, recovery, recovery_demo_ratio)


def mixed_ref_amp_body_pos_b(
    env: ManagerBasedAmpEnv,
    locomotion_animation: str,
    recovery_animation: str,
    recovery_demo_ratio: float,
    flatten_steps_dim: bool = False,
) -> torch.Tensor:
    locomotion = mdp.ref_amp_body_pos_b(env, locomotion_animation, flatten_steps_dim)
    recovery = mdp.ref_amp_body_pos_b(env, recovery_animation, flatten_steps_dim)
    return _mix_demo_tensor(env, locomotion, recovery, recovery_demo_ratio)


def mixed_ref_amp_body_lin_vel_b(
    env: ManagerBasedAmpEnv,
    locomotion_animation: str,
    recovery_animation: str,
    recovery_demo_ratio: float,
    flatten_steps_dim: bool = False,
) -> torch.Tensor:
    locomotion = mdp.ref_amp_body_lin_vel_b(env, locomotion_animation, flatten_steps_dim)
    recovery = mdp.ref_amp_body_lin_vel_b(env, recovery_animation, flatten_steps_dim)
    return _mix_demo_tensor(env, locomotion, recovery, recovery_demo_ratio)


@configclass
class MotionDataCfg:
    # motion_dataset 对应 walk/run；recovery_dataset 对应 get_up。
    motion_dataset = MotionDataTerm(motion_data_dir="", motion_data_weights={})
    recovery_dataset = MotionDataTerm(motion_data_dir="", motion_data_weights={})


@configclass
class AnimationCfg:
    # AMP discriminator 需要连续多帧参考状态；两套 animation 分别从两套 motion data 取样。
    animation = AnimTerm(
        motion_data_term=LOCOMOTION_MOTION_DATA_TERM_NAME,
        motion_data_components=[
            "root_pos_w",
            "root_quat",
            "root_vel_w",
            "root_ang_vel_w",
            "dof_pos",
            "dof_vel",
            "key_body_pos_b",
        ],
        num_steps_to_use=AMP_NUM_STEPS,
        random_initialize=True,
        random_fetch=True,
        enable_visualization=False,
    )
    recovery_animation = AnimTerm(
        motion_data_term=RECOVERY_MOTION_DATA_TERM_NAME,
        motion_data_components=[
            "root_pos_w",
            "root_quat",
            "root_vel_w",
            "root_ang_vel_w",
            "dof_pos",
            "dof_vel",
            "key_body_pos_b",
        ],
        num_steps_to_use=AMP_NUM_STEPS,
        random_initialize=True,
        random_fetch=True,
        enable_visualization=False,
    )


@configclass
class LocoRecoveryRewardsCfg:
    # 正常 locomotion 奖励：恢复窗口内会被 wrapper 置零，避免起身时还追踪速度命令。
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp_normal_only,
        weight=6.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp_normal_only,
        weight=7.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    recovery_height = RewTerm(
        func=recovery_target_base_height,
        weight=4.0,
        params={"base_height_target": 0.75, "std": 0.25},
    )
    recovery_orientation = RewTerm(
        func=recovery_target_orientation,
        weight=6.0,
        params={"std": 0.5, "standup_height_threshold": RECOVERY_ORIENTATION_HEIGHT_THRESHOLD},
    )
    recovery_low_velocity = RewTerm(
        func=recovery_low_body_velocity,
        weight=1.0,
        params={
            "lin_std": 1.0,
            "ang_std": 1.0,
            "standup_height_threshold": RECOVERY_STANDUP_HEIGHT_THRESHOLD,
        },
    )

    # --- 【局部最优克星：惩罚项隔离】 ---
    # 下列均为行走过程中的稳定性和能耗惩罚。在专属的起身(get-up)训练中，这些惩罚会被大幅调低甚至取消，
    # 因为起身需要巨大的爆发力(极大的力矩、疯狂的腿部变向)。
    # 只要这 250 步处于恢复阶段，就给机器人充分放飞自我发力的自由空间。站起并脱离掩码保护后，恢复其应有的惩罚。
    # 参考独立 GetUp 任务：由于起身需要足够的爆发力，将通用行动惩罚的权重进一步降低。
    # 相比原始 locomotion 设定（以及刚才减半的版本），这里的权重更贴近单纯 getup 的设定。
    lin_vel_z_l2 = RewTerm(func=lin_vel_z_l2_normal_only, weight=-0.05)
    ang_vel_xy_l2 = RewTerm(func=ang_vel_xy_l2_normal_only, weight=-0.0125)
    dof_torques_l2 = RewTerm(func=joint_torques_l2_normal_only, weight=-2.5e-6) # 贴合 getup: joint_torques_l2 = -2.5e-6
    dof_acc_l2 = RewTerm(func=joint_acc_l2_normal_only, weight=-2.5e-7)         # 贴合 getup: joint_acc_l2 = -2.5e-7
    action_rate_l2 = RewTerm(func=action_rate_l2_normal_only, weight=-0.01)     # 贴合 getup: action_rate_l2 = -0.01
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-100.0)          # 贴合 getup: joint_pos_limits = -100.0
    feet_air_time = RewTerm(
        func=feet_air_time_positive_biped_normal_only,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=feet_slide_normal_only,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class LocoRecoveryTerminationsCfg:
    # bad_orientation / bad_base_height 仍然会触发 termination；
    # DelayedRecoveryTerminationManager 只是在 recovery env 中暂时抑制它们。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    bad_base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})


@configclass
class G1AmpLocoRecoveryEnvCfg(G1WalkRunAmpEnvCfg):
    """G1 AMP locomotion + automatic recovery environment."""

    motion_data: MotionDataCfg = MotionDataCfg()
    animation: AnimationCfg = AnimationCfg()
    rewards: LocoRecoveryRewardsCfg = LocoRecoveryRewardsCfg()
    terminations: LocoRecoveryTerminationsCfg = LocoRecoveryTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0

        # 配置两套 pkl motion 数据源：普通走跑和摔倒起身。
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "walk_and_run"
        )
        self.motion_data.motion_dataset.motion_data_weights = _walk_run_motion_weights()
        self.motion_data.recovery_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "get_up"
        )
        self.motion_data.recovery_dataset.motion_data_weights = _recovery_motion_weights()

        self.animation.animation.motion_data_term = LOCOMOTION_MOTION_DATA_TERM_NAME
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.animation.recovery_animation.motion_data_term = RECOVERY_MOTION_DATA_TERM_NAME
        self.animation.recovery_animation.num_steps_to_use = AMP_NUM_STEPS

        # startup 时安装延迟终止 manager。delay_env_ratio 控制多少并行环境参与恢复训练；
        # max_delay_steps 控制摔倒后最多给多少步尝试恢复。
        self.events.install_recovery_delay = EventTerm(
            func=install_delayed_recovery_termination,
            mode="startup",
            params={
                "delay_env_ratio": 0.35,
                "max_delay_steps": 300,
            },
        )
        # reset 时根据 env 是否属于 recovery 子集，从不同 motion dataset 初始化。
        self.events.reset_from_ref.func = reset_from_locomotion_or_recovery
        self.events.reset_from_ref.params = {
            "locomotion_motion_data_term": LOCOMOTION_MOTION_DATA_TERM_NAME,
            "recovery_motion_data_term": RECOVERY_MOTION_DATA_TERM_NAME,
            "height_offset": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        }

        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        # Get-up uses an upward external force to reduce early exploration difficulty.
        # In the mixed locomotion/recovery task, apply it only during active recovery
        # and fade it out as the torso approaches the standing target height.
        self.events.recovery_upward_assist = EventTerm(
            func=apply_recovery_upward_assist,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "force_magnitude": RECOVERY_ASSIST_FORCE_N,
                "fade_start_height": RECOVERY_STANDUP_HEIGHT_THRESHOLD,
                "fade_end_height": RECOVERY_ASSIST_FADE_END_HEIGHT,
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        )
        # 间歇推搡机器人，制造跌倒/失稳样本，触发恢复窗口。
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            params={
                "velocity_range": {
                    "x": (-0.6, 0.6),
                    "y": (-0.3, 0.3),
                    "z": (-0.2, 0.2),
                    "roll": (-0.25, 0.25),
                    "pitch": (-0.25, 0.25),
                    "yaw": (-0.4, 0.4),
                }
            },
        )

        # discriminator demo 混合一部分 get_up 参考，使 AMP 不只模仿走跑，也见过起身动作。
        demo_params = {
            "locomotion_animation": LOCOMOTION_ANIMATION_TERM_NAME,
            "recovery_animation": RECOVERY_ANIMATION_TERM_NAME,
            "recovery_demo_ratio": 0.4,
            "flatten_steps_dim": False,
        }
        self.observations.disc_demo.ref_root_local_rot_tan_norm.func = mixed_ref_root_local_rot_tan_norm
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params = demo_params.copy()
        self.observations.disc_demo.ref_root_ang_vel_b.func = mixed_ref_root_ang_vel_b
        self.observations.disc_demo.ref_root_ang_vel_b.params = demo_params.copy()
        self.observations.disc_demo.ref_body_pos_b.func = mixed_ref_amp_body_pos_b
        self.observations.disc_demo.ref_body_pos_b.params = demo_params.copy()
        self.observations.disc_demo.ref_body_lin_vel_b.func = mixed_ref_amp_body_lin_vel_b
        self.observations.disc_demo.ref_body_lin_vel_b.params = demo_params.copy()

        # Bring AMP observations closer to mjlab's body-space discriminator
        # features. The pkl pipeline stores key-body positions but not full
        # per-body orientations, so we use root-frame key-body positions and
        # relative key-body linear velocities alongside root orientation/ang vel.
        amp_body_cfg = SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        self.observations.disc.joint_pos = None
        self.observations.disc.joint_vel = None
        self.observations.disc.key_body_pos_b = None
        self.observations.disc.body_pos_b = ObsTerm(
            func=mdp.amp_body_pos_b,
            params={"asset_cfg": amp_body_cfg},
        )
        self.observations.disc.body_lin_vel_b = ObsTerm(
            func=mdp.amp_body_lin_vel_b,
            params={"asset_cfg": amp_body_cfg},
        )

        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        # 扩大命令范围并关闭 curriculum，直接覆盖完整速度命令分布。
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None


@configclass
class G1AmpLocoRecoveryEnvCfg_PLAY(G1AmpLocoRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # PLAY 配置用于可视化/调试：少量环境、超长 episode、无观测噪声、无随机推搡。
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.recovery_upward_assist = None
        self.events.install_recovery_delay.params["delay_env_ratio"] = 1.0


@configclass
class G1AmpMultiCriticLocoRecoveryEnvCfg(G1AmpLocoRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Multi-critic semantics:
        # - task: locomotion and recovery objectives;
        # - regularization: safety, termination and smoothness penalties;
        # - style: filled by MultiCriticPPOAMP with the AMP discriminator reward.
        self.reward_groups = {
            "task": [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "feet_air_time",
                "recovery_height",
                "recovery_orientation",
                "recovery_low_velocity",
            ],
            "regularization": [
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "dof_pos_limits",
                "feet_slide",
                "termination_penalty",
            ],
            "style": [],
        }


@configclass
class G1AmpMultiCriticLocoRecoveryEnvCfg_PLAY(G1AmpMultiCriticLocoRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 多 critic 的 PLAY 配置同样用于调试：全量恢复环境、关闭噪声和随机推搡。
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.recovery_upward_assist = None
        self.events.install_recovery_delay.params["delay_env_ratio"] = 1.0
