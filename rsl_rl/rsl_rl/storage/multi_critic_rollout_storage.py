from __future__ import annotations

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage


class MultiCriticRolloutStorage(RolloutStorage):
    """Rollout storage where reward, value, return and advantage carry a critic dimension."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs,
        actions_shape: tuple[int] | list[int],
        num_critics: int,
        device: str = "cpu",
    ) -> None:
        if num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {num_critics}.")
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        if training_type != "rl":
            raise ValueError("MultiCriticRolloutStorage only supports reinforcement learning storage.")

        self.num_critics = num_critics
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(self.num_envs, self.num_critics))
        self.dones[self.step].copy_(transition.dones.view(self.num_envs, 1))
        self.values[self.step].copy_(transition.values.view(self.num_envs, self.num_critics))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(self.num_envs, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1
