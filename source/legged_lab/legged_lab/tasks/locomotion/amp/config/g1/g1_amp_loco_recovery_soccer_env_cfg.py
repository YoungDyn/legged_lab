"""RoboCup HSL L-Field soccer variants for G1 AMP loco-recovery.

This module intentionally keeps the soccer field setup isolated from
``g1_amp_loco_recovery_env_cfg.py``. The base locomotion/recovery task remains
unchanged, while these variants add the field, FIFA size 5 ball, soccer
observations, and Wang2025-style reward grouping.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import legged_lab.tasks.locomotion.amp.mdp as mdp
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.robocup_hsl_l_field import (
    FIELD_LENGTH,
    GOAL_HEIGHT,
    GOAL_WIDTH,
    spawn_robocup_hsl_l_field,
)

from .g1_amp_loco_recovery_env_cfg import (
    G1AmpLocoRecoveryEnvCfg_PLAY,
    G1AmpMultiCriticLocoRecoveryEnvCfg_PLAY,
    LocoRecoveryRewardsCfg,
)


SOCCER_BALL_RADIUS_M = 0.11
"""FIFA size 5 ball radius. Size 5 circumference is about 0.68-0.70 m."""

SOCCER_BALL_MASS_KG = 0.43
"""FIFA size 5 ball mass range is 0.410-0.450 kg; use the midpoint."""

RIGHT_GOAL_X = FIELD_LENGTH * 0.5
SOCCER_GOAL_POS_W = (RIGHT_GOAL_X, 0.0)
SOCCER_BALL_REACH_DISTANCE = 0.45
SOCCER_FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
SOCCER_NON_FOOT_CONTACT_BODY_PATTERN = "(?!.*ankle_roll_link).*"


class SoccerRewardState:
    """Per-env state needed by soccer differential rewards."""

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.prev_robot_ball_dist = torch.zeros(num_envs, device=device)
        self.prev_ball_goal_dist = torch.zeros(num_envs, device=device)
        self.stagnation_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_base_lin_vel_w = torch.zeros((num_envs, 3), device=device)

    def reset(
        self,
        env_ids: torch.Tensor,
        robot_ball_dist: torch.Tensor,
        ball_goal_dist: torch.Tensor,
        base_lin_vel_w: torch.Tensor,
    ) -> None:
        self.prev_robot_ball_dist[env_ids] = robot_ball_dist[env_ids].detach()
        self.prev_ball_goal_dist[env_ids] = ball_goal_dist[env_ids].detach()
        self.stagnation_counter[env_ids] = 0
        self.prev_base_lin_vel_w[env_ids] = base_lin_vel_w[env_ids].detach()

    def robot_ball_progress(self, current: torch.Tensor) -> torch.Tensor:
        reward = self.prev_robot_ball_dist - current
        self.prev_robot_ball_dist.copy_(current.detach())
        return reward

    def ball_goal_progress(self, current: torch.Tensor) -> torch.Tensor:
        reward = self.prev_ball_goal_dist - current
        self.prev_ball_goal_dist.copy_(current.detach())
        return reward

    def base_acceleration_l2(self, current_vel_w: torch.Tensor, dt: float) -> torch.Tensor:
        acceleration = (current_vel_w - self.prev_base_lin_vel_w) / dt
        self.prev_base_lin_vel_w.copy_(current_vel_w.detach())
        return torch.sum(torch.square(acceleration), dim=-1)


def _soccer_reward_state(env) -> SoccerRewardState:
    state = getattr(env, "_soccer_reward_state", None)
    if not isinstance(state, SoccerRewardState) or state.prev_robot_ball_dist.shape != (env.num_envs,):
        state = SoccerRewardState(env.num_envs, env.device)
        env._soccer_reward_state = state
    return state


def _robot_and_ball(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> tuple[Articulation, RigidObject]:
    return env.scene[robot_cfg.name], env.scene[ball_cfg.name]


def _goal_pos_w(env, goal_pos_w: tuple[float, float] = SOCCER_GOAL_POS_W) -> torch.Tensor:
    goal = torch.tensor((goal_pos_w[0], goal_pos_w[1], SOCCER_BALL_RADIUS_M), device=env.device)
    return env.scene.env_origins + goal


def _yaw_frame_xy(robot: Articulation, rel_w: torch.Tensor) -> torch.Tensor:
    rel_w = rel_w.clone()
    rel_w[:, 2] = 0.0
    return math_utils.quat_apply_inverse(math_utils.yaw_quat(robot.data.root_quat_w), rel_w)[:, :2]


def _ball_pos_robot_xy(env, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    robot, ball = _robot_and_ball(env, robot_cfg, ball_cfg)
    return _yaw_frame_xy(robot, ball.data.root_pos_w - robot.data.root_pos_w)


def _robot_ball_distance(env, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    robot, ball = _robot_and_ball(env, robot_cfg, ball_cfg)
    return torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)


def _ball_goal_distance(
    env,
    ball_cfg: SceneEntityCfg,
    goal_pos_w: tuple[float, float] = SOCCER_GOAL_POS_W,
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    goal_w = _goal_pos_w(env, goal_pos_w)
    return torch.norm(goal_w[:, :2] - ball.data.root_pos_w[:, :2], dim=-1)


def _reset_soccer_reward_state(env, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> None:
    current_robot_ball = _robot_ball_distance(env, robot_cfg, ball_cfg)
    current_ball_goal = _ball_goal_distance(env, ball_cfg)
    current_vel = env.scene[robot_cfg.name].data.root_lin_vel_w
    _soccer_reward_state(env).reset(env_ids, current_robot_ball, current_ball_goal, current_vel)


def initialize_soccer_reward_state(
    env,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Pre-allocate soccer reward state and align all envs with current scene state."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    _reset_soccer_reward_state(env, env_ids, robot_cfg, ball_cfg)


def reset_soccer_ball(
    env,
    env_ids: torch.Tensor,
    distance_range: tuple[float, float],
    angle_range: tuple[float, float],
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset a FIFA size 5 ball around the robot in the robot yaw frame."""

    if env_ids is None or len(env_ids) == 0:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    num_resets = len(env_ids)

    rel_b = torch.zeros((num_resets, 3), device=env.device)
    distance = torch.empty(num_resets, device=env.device).uniform_(*distance_range)
    angle = torch.empty(num_resets, device=env.device).uniform_(*angle_range)
    rel_b[:, 0] = distance * torch.cos(angle)
    rel_b[:, 1] = distance * torch.sin(angle)
    rel_w = math_utils.quat_apply(math_utils.yaw_quat(robot.data.root_quat_w[env_ids]), rel_b)
    ball_pos = robot.data.root_pos_w[env_ids] + rel_w
    ball_pos[:, 2] = SOCCER_BALL_RADIUS_M

    root_state = torch.zeros((num_resets, 13), device=env.device)
    root_state[:, :3] = ball_pos
    root_state[:, 3] = 1.0
    ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    _reset_soccer_reward_state(env, env_ids, robot_cfg, ball_cfg)


def soccer_ball_observation(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Actor-visible perfect ball perception: x/y in robot yaw frame + mask."""

    ball_xy = _ball_pos_robot_xy(env, robot_cfg, ball_cfg)
    mask = torch.ones((env.num_envs, 1), device=env.device)
    return torch.cat((ball_xy, mask), dim=-1)


def soccer_goal_observation(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_pos_w: tuple[float, float] = SOCCER_GOAL_POS_W,
) -> torch.Tensor:
    """Goal position and direction in robot yaw frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    goal_xy = _yaw_frame_xy(robot, _goal_pos_w(env, goal_pos_w) - robot.data.root_pos_w)
    direction = goal_xy / goal_xy.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    return torch.cat((goal_xy, direction), dim=-1)


def soccer_ball_velocity_privileged(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Critic-only ball velocity in the robot yaw frame."""

    robot, ball = _robot_and_ball(env, robot_cfg, ball_cfg)
    rel_vel_w = ball.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return _yaw_frame_xy(robot, rel_vel_w)


def soccer_base_height(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_pos_w[:, 2:3]


def soccer_zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def soccer_survival(env) -> torch.Tensor:
    terminated = getattr(env.termination_manager, "terminated", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return (~terminated).float()


def soccer_goal_scored(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    goal_x: float = RIGHT_GOAL_X,
    goal_width: float = GOAL_WIDTH,
    goal_height: float = GOAL_HEIGHT,
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    goal_x_w = env.scene.env_origins[:, 0] + goal_x
    goal_y_w = env.scene.env_origins[:, 1]
    crossed_line = ball.data.root_pos_w[:, 0] >= goal_x_w
    inside_mouth = (ball.data.root_pos_w[:, 1] - goal_y_w).abs() <= goal_width * 0.5
    under_bar = ball.data.root_pos_w[:, 2] <= goal_height
    return (crossed_line & inside_mouth & under_bar).float()


def soccer_ball_approach(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    current = _robot_ball_distance(env, robot_cfg, ball_cfg)
    return _soccer_reward_state(env).robot_ball_progress(current)


def soccer_goal_progress(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    goal_pos_w: tuple[float, float] = SOCCER_GOAL_POS_W,
) -> torch.Tensor:
    current = _ball_goal_distance(env, ball_cfg, goal_pos_w)
    return _soccer_reward_state(env).ball_goal_progress(current)


def soccer_stagnation(
    env,
    speed_threshold: float,
    duration_s: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    state = _soccer_reward_state(env)
    slow = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1) < speed_threshold
    counter = torch.where(slow, state.stagnation_counter + 1, torch.zeros_like(state.stagnation_counter))
    state.stagnation_counter.copy_(counter)
    steps = max(1, int(duration_s / env.step_dt))
    return (counter >= steps).float()


def soccer_sideways_kick(
    env,
    contact_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=SOCCER_FOOT_BODY_NAMES),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot, ball = _robot_and_ball(env, robot_cfg, ball_cfg)
    feet_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    feet_vel_w = robot.data.body_lin_vel_w[:, robot_cfg.body_ids, :]
    ball_pos_w = ball.data.root_pos_w[:, None, :]
    near = torch.norm(feet_pos_w[:, :, :2] - ball_pos_w[:, :, :2], dim=-1) < contact_distance
    yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)
    lateral_speeds = []
    for i in range(feet_vel_w.shape[1]):
        foot_vel_b = math_utils.quat_apply_inverse(yaw_quat, feet_vel_w[:, i, :])
        lateral_speeds.append(foot_vel_b[:, 1].abs() * near[:, i].float())
    return torch.stack(lateral_speeds, dim=-1).max(dim=-1).values


def soccer_forward_kick(
    env,
    contact_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=SOCCER_FOOT_BODY_NAMES),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot, ball = _robot_and_ball(env, robot_cfg, ball_cfg)
    feet_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    feet_vel_w = robot.data.body_lin_vel_w[:, robot_cfg.body_ids, :]
    ball_pos_w = ball.data.root_pos_w[:, None, :]
    near = torch.norm(feet_pos_w[:, :, :2] - ball_pos_w[:, :, :2], dim=-1) < contact_distance
    yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)
    forward_speeds = []
    for i in range(feet_vel_w.shape[1]):
        foot_vel_b = math_utils.quat_apply_inverse(yaw_quat, feet_vel_w[:, i, :])
        forward_speeds.append(foot_vel_b[:, 0].abs() * near[:, i].float())
    return torch.stack(forward_speeds, dim=-1).max(dim=-1).values


def soccer_foot_proximity(
    env,
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=SOCCER_FOOT_BODY_NAMES),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    feet_pos = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    return (threshold - torch.norm(feet_pos[:, 0, :] - feet_pos[:, 1, :], dim=-1)).clamp_min(0.0)


def soccer_base_acceleration(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    current = robot.data.root_lin_vel_w
    return _soccer_reward_state(env).base_acceleration_l2(current, env.step_dt)


@configclass
class SoccerRewardsCfg(LocoRecoveryRewardsCfg):
    """Wang2025-style soccer rewards, isolated from the loco-recovery base."""

    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    recovery_height = None
    recovery_orientation = None
    recovery_low_velocity = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    dof_torques_l2 = None
    dof_acc_l2 = None
    action_rate_l2 = None
    feet_air_time = None
    feet_slide = None

    survival = RewTerm(func=soccer_survival, weight=3.0)
    goal_scored = RewTerm(func=soccer_goal_scored, weight=15.0)
    ball_approach = RewTerm(func=soccer_ball_approach, weight=50.0)
    goal_progress = RewTerm(func=soccer_goal_progress, weight=500.0)
    stagnation = RewTerm(func=soccer_stagnation, weight=-100.0, params={"speed_threshold": 0.05, "duration_s": 1.0})
    head_pitch_alignment = RewTerm(func=soccer_zero_reward, weight=-0.5)
    head_yaw_alignment = RewTerm(func=soccer_zero_reward, weight=-0.5)
    sideways_kick = RewTerm(func=soccer_sideways_kick, weight=20.0, params={"contact_distance": SOCCER_BALL_REACH_DISTANCE})
    forward_kick = RewTerm(func=soccer_forward_kick, weight=-20.0, params={"contact_distance": SOCCER_BALL_REACH_DISTANCE})
    foot_proximity = RewTerm(func=soccer_foot_proximity, weight=-5.0, params={"threshold": 0.18})
    head_action_rate = RewTerm(func=soccer_zero_reward, weight=-15.0)
    leg_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-100.0)
    base_acceleration = RewTerm(func=soccer_base_acceleration, weight=-0.001)
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-100.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=SOCCER_NON_FOOT_CONTACT_BODY_PATTERN), "threshold": 1.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000.0)


@configclass
class G1AmpLocoRecoverySoccerEnvCfg_PLAY(G1AmpLocoRecoveryEnvCfg_PLAY):
    """Single-G1 PLAY config on a RoboCup HSL Large L-Field."""

    rewards: SoccerRewardsCfg = SoccerRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SoccerBall",
            spawn=sim_utils.SphereCfg(
                radius=SOCCER_BALL_RADIUS_M,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_linear_velocity=40.0,
                    max_angular_velocity=200.0,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=SOCCER_BALL_MASS_KG),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=0.45,
                    dynamic_friction=0.35,
                    restitution=0.55,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.90), roughness=0.6),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, SOCCER_BALL_RADIUS_M)),
        )

        self.scene.num_envs = 1
        self.scene.env_spacing = 24.0
        self.viewer.eye = (0.0, -18.0, 14.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.events.spawn_robocup_hsl_l_field = EventTerm(
            func=spawn_robocup_hsl_l_field,
            mode="startup",
            params={},
        )
        self.events.initialize_soccer_reward_state = EventTerm(
            func=initialize_soccer_reward_state,
            mode="startup",
            params={},
        )
        self.events.reset_soccer_ball = EventTerm(
            func=reset_soccer_ball,
            mode="reset",
            params={"distance_range": (0.8, 3.0), "angle_range": (-math.pi, math.pi)},
        )

        self.observations.policy.soccer_ball = ObsTerm(func=soccer_ball_observation)
        self.observations.policy.soccer_goal = ObsTerm(func=soccer_goal_observation)
        self.observations.critic.soccer_ball = ObsTerm(func=soccer_ball_observation)
        self.observations.critic.soccer_goal = ObsTerm(func=soccer_goal_observation)
        self.observations.critic.soccer_ball_velocity = ObsTerm(func=soccer_ball_velocity_privileged)
        self.observations.critic.soccer_base_height = ObsTerm(func=soccer_base_height)


@configclass
class G1AmpMultiCriticLocoRecoverySoccerEnvCfg_PLAY(G1AmpMultiCriticLocoRecoveryEnvCfg_PLAY):
    """Multi-critic PLAY config on a RoboCup HSL Large L-Field."""

    rewards: SoccerRewardsCfg = SoccerRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SoccerBall",
            spawn=sim_utils.SphereCfg(
                radius=SOCCER_BALL_RADIUS_M,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_linear_velocity=40.0,
                    max_angular_velocity=200.0,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=SOCCER_BALL_MASS_KG),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=0.45,
                    dynamic_friction=0.35,
                    restitution=0.55,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.90), roughness=0.6),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, SOCCER_BALL_RADIUS_M)),
        )

        self.scene.num_envs = 1
        self.scene.env_spacing = 24.0
        self.viewer.eye = (0.0, -18.0, 14.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.events.spawn_robocup_hsl_l_field = EventTerm(
            func=spawn_robocup_hsl_l_field,
            mode="startup",
            params={},
        )
        self.events.initialize_soccer_reward_state = EventTerm(
            func=initialize_soccer_reward_state,
            mode="startup",
            params={},
        )
        self.events.reset_soccer_ball = EventTerm(
            func=reset_soccer_ball,
            mode="reset",
            params={"distance_range": (0.8, 3.0), "angle_range": (-math.pi, math.pi)},
        )

        self.observations.policy.soccer_ball = ObsTerm(func=soccer_ball_observation)
        self.observations.policy.soccer_goal = ObsTerm(func=soccer_goal_observation)
        self.observations.critic.soccer_ball = ObsTerm(func=soccer_ball_observation)
        self.observations.critic.soccer_goal = ObsTerm(func=soccer_goal_observation)
        self.observations.critic.soccer_ball_velocity = ObsTerm(func=soccer_ball_velocity_privileged)
        self.observations.critic.soccer_base_height = ObsTerm(func=soccer_base_height)
        self.reward_groups = {
            "task": [
                "survival",
                "goal_scored",
                "ball_approach",
                "goal_progress",
                "sideways_kick",
                "forward_kick",
                "foot_proximity",
            ],
            "regularization": [
                "termination_penalty",
                "stagnation",
                "head_pitch_alignment",
                "head_yaw_alignment",
                "head_action_rate",
                "leg_action_rate",
                "dof_pos_limits",
                "base_acceleration",
                "collision",
            ],
            "style": [],
        }
