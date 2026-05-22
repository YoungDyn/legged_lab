from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP


class MultiCriticActorCritic(ActorCritic):
    """ActorCritic variant with one shared actor and K independent critic heads."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_critics: int = 1,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs,
    ) -> None:
        if num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {num_critics}.")

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )

        self.num_critics = num_critics
        num_critic_obs = sum(obs[obs_group].shape[-1] for obs_group in obs_groups["critic"])
        self.critics = torch.nn.ModuleList(
            [MLP(num_critic_obs, 1, critic_hidden_dims, activation) for _ in range(num_critics)]
        )
        del self.critic
        print(f"MultiCritic critics ({num_critics}): {self.critics}")

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return torch.cat([critic(critic_obs) for critic in self.critics], dim=-1)
