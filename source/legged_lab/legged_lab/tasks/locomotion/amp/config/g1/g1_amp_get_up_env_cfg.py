import os
import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
] # if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.g1 as well!
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4

class RobotAsset:
    base_name = 'torso_link'

    target_base_height_phase3 = 0.65

    base_height_target = 0.75


@configclass
class CommandsCfg:
    force_command = mdp.ForceCommandCfg(
        force=0.0,
        resampling_time_range=[100.0, 100.0]
    )

@configclass
class ObservationsCfg():
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
            noise=Unoise(n_min=-0.04, n_max=0.04),
        )

        # root_height = ObsTerm(func=mdp.base_pos_z)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class DiscriminatorCfg(ObsGroup):
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 10
            self.flatten_history_dim = False

    disc: DiscriminatorCfg = DiscriminatorCfg()

    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        ref_root_local_rot_tan_norm = ObsTerm(
            func=mdp.ref_root_local_rot_tan_norm,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()

@configclass
class EventsCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[RobotAsset.base_name]),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    apply_force = EventTerm(
        func=mdp.apply_force,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[RobotAsset.base_name]),
        }
    )

    reset_from_ref = EventTerm(
        func=mdp.reset_from_ref,
        mode="reset",
        params=MISSING
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.3, -0.3),
                "roll": (-torch.pi, torch.pi),
                "pitch": (-torch.pi / 2, torch.pi / 2),
                "yaw": (-torch.pi, torch.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.02
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-6
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-100.0
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy,
        weight=3.0,
        params={
            "target_base_height_phase3": RobotAsset.target_base_height_phase3,
            "asset_cfg": SceneEntityCfg("robot")
        }
    )
    lin_vel_xy = RewTerm(
        func=mdp.lin_vel_xy,
        weight=3.0,
        params={
            "target_base_height_phase3": RobotAsset.target_base_height_phase3,
            "asset_cfg": SceneEntityCfg("robot")
        }
    )
    target_orientation = RewTerm(
        func=mdp.target_orientation,
        weight=3.0,
        params={
            "target_base_height_phase3": RobotAsset.target_base_height_phase3,
            "asset_cfg": SceneEntityCfg("robot")
        }
    )
    target_base_height = RewTerm(
        func=mdp.target_base_height,
        weight=5.0,
        params={
            "base_height_target": RobotAsset.base_height_target,
            "target_base_height_phase3": RobotAsset.target_base_height_phase3,
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 300.0
        }
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    force_level = CurrTerm(
        func=mdp.force_level,
        params={
            "reward_term_name": "target_base_height"
        }
    )

@configclass
class G1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the G1 AMP environment."""

    observations = ObservationsCfg()
    commands = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "get_up"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "fallAndGetUp1_subject1_680_800": 0.0,
            "fallAndGetUp1_subject1_850_940": 0.0,
            "fallAndGetUp1_subject1_1060_1150": 0.0,
            "fallAndGetUp1_subject1_1400_1480": 0.0,
            "fallAndGetUp1_subject1_1600_1700": 0.0,
            "fallAndGetUp1_subject1_2100_2200": 0.0,
            "fallAndGetUp1_subject1_2300_2400": 0.0,
            "fallAndGetUp1_subject4_3700_3800": 0.0,
            "fallAndGetUp1_subject5_2500_2600": 0.0,
            "fallAndGetUp2_subject2_360_580": 0.0,
            "fallAndGetUp2_subject2_850_1050": 0.0,
            "fallAndGetUp2_subject2_1200_1370": 1.0,
            "fallAndGetUp2_subject2_1500_1600": 0.0,
            "fallAndGetUp2_subject3_900_1000": 0.0,
            "fallAndGetUp2_subject3_1850_1920": 0.0,
            "fallAndGetUp2_subject3_2080_2180": 0.0,
            "fallAndGetUp6_subject1_530_600": 0.0,
            "fallAndGetUp6_subject1_650_700": 0.0,
            "fallAndGetUp6_subject1_1080_1180": 0.0,
            "fallAndGetUp6_subject1_1230_1300": 0.0,
            "fallAndGetUp6_subject1_1630_1690": 0.0,
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------

        # policy observations

        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True
            )
        }

        # critic observations

        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True
            )
        }

        # discriminator observations

        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True
            )
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        # discriminator demostration observations

        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        # self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        # # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        # self.events.reset_from_ref.params = {
        #     "animation": ANIMATION_TERM_NAME,
        #     "height_offset": 0.1
        # }
        # 已经能基本起来，关闭从参考动作重置，改为随机初始化 root 姿态和 dof_vel。
        self.events.reset_from_ref = None

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        # self.curriculum.lin_vel_cmd_levels = None
        # self.curriculum.ang_vel_cmd_levels = None

        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        # self.terminations.base_contact = None


@configclass
class G1AmpEnvCfg_PLAY(G1AmpEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.force_command.force = 0.0
        self.events.apply_force = None
        self.events.reset_from_ref = None
        self.events.push_robot = None
        self.curriculum.force_level = None


@configclass
class G1AmpMultiCriticGetUpEnvCfg(G1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Define getup and aux reward groups per the instructions
        self.reward_groups = {
            "getup": [
                "ang_vel_xy", "lin_vel_xy", "target_orientation", "target_base_height"
            ],
            "aux": [
                "joint_acc_l2", "action_rate_l2", "joint_torques_l2", "joint_pos_limits"
            ]
        }

@configclass
class G1AmpMultiCriticGetUpEnvCfg_PLAY(G1AmpMultiCriticGetUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.force_command.force = 0.0
        self.events.apply_force = None
        self.events.reset_from_ref = None
        self.events.push_robot = None
        self.curriculum.force_level = None
