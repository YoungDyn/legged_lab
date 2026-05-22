from __future__ import annotations

import warnings

from tensordict import TensorDict

from rsl_rl.algorithms import PPOAMP
from rsl_rl.algorithms.multi_critic_ppo_amp import MultiCriticPPOAMP
from rsl_rl.modules import (
    resolve_amp_config,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.modules.multi_critic_actor_critic import MultiCriticActorCritic
from rsl_rl.runners.amp_runner import AMPRunner
from rsl_rl.storage import CircularBuffer
from rsl_rl.storage.multi_critic_rollout_storage import MultiCriticRolloutStorage


class MultiCriticAMPRunner(AMPRunner):
    """AMP runner that wires MultiCriticActorCritic, MultiCriticRolloutStorage and MultiCriticPPOAMP."""

    alg: MultiCriticPPOAMP

    def _construct_algorithm(self, obs: TensorDict) -> PPOAMP:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        self.alg_cfg = resolve_amp_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        if "critic_nums" in self.alg_cfg and "num_critics" not in self.alg_cfg:
            self.alg_cfg["num_critics"] = self.alg_cfg["critic_nums"]
        num_critics = self.alg_cfg.get("num_critics", 1)

        policy_class_name = self.policy_cfg.pop("class_name", "MultiCriticActorCritic")
        if policy_class_name not in {"ActorCritic", "MultiCriticActorCritic"}:
            raise ValueError(
                "MultiCriticAMPRunner currently supports feed-forward ActorCritic policies only, "
                f"got {policy_class_name!r}."
            )
        actor_critic = MultiCriticActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            num_critics=num_critics,
            **self.policy_cfg,
        ).to(self.device)

        storage = MultiCriticRolloutStorage(
            "rl",
            self.env.num_envs,
            self.cfg["num_steps_per_env"],
            obs,
            [self.env.num_actions],
            num_critics=num_critics,
            device=self.device,
        )

        disc_obs_buffer = CircularBuffer(
            max_len=self.alg_cfg["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=self.env.num_envs,
            device=self.device,
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=self.alg_cfg["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=self.env.num_envs,
            device=self.device,
        )

        alg_class_name = self.alg_cfg.pop("class_name")
        if alg_class_name != "MultiCriticPPOAMP":
            raise ValueError(f"MultiCriticAMPRunner requires MultiCriticPPOAMP, got {alg_class_name!r}.")
        alg = MultiCriticPPOAMP(
            actor_critic,
            storage,
            disc_obs_buffer,
            disc_demo_obs_buffer,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        return alg
