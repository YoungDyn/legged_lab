import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg
from legged_lab.rsl_rl import RslRlPpoAmpAlgorithmCfg, RslRlAmpCfg, RslRlPpoActorCriticConv2dCfg
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.mdp.symmetry import g1

@configclass
class G1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "g1_amp"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"]
    }
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     actor_obs_normalization=False,
    #     critic_obs_normalization=False,
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=64,
    #     rnn_num_layers=1
    # )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        min_policy_std=0.05,
        max_policy_std=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            # Match the more conservative mj_lab discriminator regularization.
            # The discriminator has its own fixed optimizer here, while the PPO
            # policy LR can be reduced by the adaptive KL schedule.
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512, 256],
                activation="elu",
                style_reward_scale=5.0,
                # mj_lab uses amp_reward_coef=0.1 with task_reward_lerp=0.75.
                # With dt-scaled style rewards, this keeps the policy less
                # dominated by a fast discriminator early in training.
                task_style_lerp=0.75
            ),
            loss_type="LSGAN"
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True, mirror_loss_coeff=0.1,
        )
    )


@configclass
class RslRlMultiCriticPpoAmpAlgorithmCfg(RslRlPpoAmpAlgorithmCfg):
    num_critics: int = 3
    critic_names: list[str] = ["task", "regularization", "style"]
    reward_group_weights: list[float] = [1.0, 0.5, 0.25]
    value_loss_weights: list[float] = [1.0, 0.25, 1.0]
    normalize_advantage_per_critic: bool = True
    critic_type: str = "independent"
    amp: dict = None


@configclass
class G1MultiCriticLocoRecoveryRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    class_name = "MultiCriticAMPRunner"
    experiment_name = "g1_multicritic_amp_loco_recovery"
    algorithm = RslRlMultiCriticPpoAmpAlgorithmCfg(
        class_name="MultiCriticPPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        min_policy_std=0.05,
        max_policy_std=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            # Match the more conservative mj_lab discriminator regularization.
            # The discriminator has its own fixed optimizer here, while the PPO
            # policy LR can be reduced by the adaptive KL schedule.
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512, 256],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.75
            ),
            loss_type="LSGAN"
        ),
        num_critics=3,
        critic_names=["task", "regularization", "style"],
        # actor advantage = task + 0.5 * regularization + 0.25 * style
        reward_group_weights=[1.0, 0.5, 0.25],
        # Regularization has much larger negative returns; down-weight its
        # critic loss so it does not dominate the policy/value update.
        value_loss_weights=[1.0, 0.25, 1.0],
        normalize_advantage_per_critic=True,
        critic_type="independent",
        amp={
            "reward_group": "style",
            # Style has its own critic. With style_reward_scale=5 and dt=0.02,
            # the max style reward is about 0.1; the 0.25 group weight matches
            # mj_lab's 0.1 * (1 - 0.75) effective scale.
            "amp_reward_weight": 1.0,
        },
    )


@configclass
class G1MultiCriticSoccerRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    class_name = "MultiCriticAMPRunner"
    experiment_name = "g1_multicritic_amp_loco_recovery_soccer"
    algorithm = RslRlMultiCriticPpoAmpAlgorithmCfg(
        class_name="MultiCriticPPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512, 256],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.75,
            ),
            loss_type="LSGAN",
        ),
        num_critics=3,
        critic_names=["task", "regularization", "style"],
        reward_group_weights=[2.0, 0.5, 0.25],
        value_loss_weights=[1.0, 0.25, 1.0],
        normalize_advantage_per_critic=True,
        critic_type="independent",
        amp={
            "reward_group": "style",
            "amp_reward_weight": 1.0,
        },
    )
