from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO


class MultiCriticPPO(PPO):
    """PPO with K independent value functions and a weighted actor advantage."""

    def __init__(
        self,
        policy,
        storage,
        num_critics: int = 1,
        critic_names: list[str] | None = None,
        reward_group_weights: list[float] | None = None,
        value_loss_weights: list[float] | None = None,
        normalize_advantage_per_critic: bool = True,
        critic_type: str = "independent",
        use_multi_critic: bool | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        if "critic_nums" in kwargs and num_critics == 1:
            num_critics = kwargs.pop("critic_nums")
        super().__init__(policy, storage, device=device, **kwargs)

        if critic_type != "independent":
            raise ValueError(f"Only independent critics are supported, got critic_type={critic_type!r}.")

        self.num_critics = num_critics
        self.critic_names = critic_names or [f"critic_{i}" for i in range(num_critics)]
        reward_group_weights = reward_group_weights or [1.0] * num_critics
        value_loss_weights = value_loss_weights or [1.0] * num_critics
        self._validate_multi_critic_config(reward_group_weights, value_loss_weights)

        self.reward_group_weights = torch.tensor(reward_group_weights, device=device, dtype=torch.float32).view(1, -1)
        self.value_loss_weights = torch.tensor(value_loss_weights, device=device, dtype=torch.float32).view(-1)
        self.normalize_advantage_per_critic = normalize_advantage_per_critic

    def _validate_multi_critic_config(self, reward_group_weights: list[float], value_loss_weights: list[float]) -> None:
        if self.num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {self.num_critics}.")
        if len(self.critic_names) != self.num_critics:
            raise ValueError(f"critic_names length must be {self.num_critics}, got {len(self.critic_names)}.")
        if len(set(self.critic_names)) != len(self.critic_names):
            raise ValueError(f"critic_names must be unique, got {self.critic_names}.")
        if len(reward_group_weights) != self.num_critics:
            raise ValueError(f"reward_group_weights length must be {self.num_critics}.")
        if len(value_loss_weights) != self.num_critics:
            raise ValueError(f"value_loss_weights length must be {self.num_critics}.")

    def _get_reward_groups(self, rewards: torch.Tensor, extras: dict) -> torch.Tensor:
        if "reward_groups" in extras:
            reward_groups = extras["reward_groups"]
        elif "extras" in extras and "reward_groups" in extras["extras"]:
            reward_groups = extras["extras"]["reward_groups"]
        elif self.num_critics == 1:
            reward_groups = rewards.view(-1, 1)
        else:
            raise RuntimeError(
                "MultiCriticPPO requires extras['reward_groups'] with shape [num_envs, num_critics]."
            )
        reward_groups = reward_groups.to(self.device)
        if reward_groups.shape != (self.storage.num_envs, self.num_critics):
            raise RuntimeError(
                "Invalid reward_groups shape: "
                f"expected {(self.storage.num_envs, self.num_critics)}, got {tuple(reward_groups.shape)}."
            )
        return reward_groups

    def process_env_step(self, obs, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = self._get_reward_groups(rewards, extras).clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards.view(-1, 1)

        if "time_outs" in extras:
            timeouts = extras["time_outs"].view(-1, 1).to(self.device)
            self.transition.rewards += self.gamma * self.transition.values * timeouts

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs) -> None:
        st = self.storage
        last_values = self.policy.evaluate(obs).detach()
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            if self.normalize_advantage_per_critic:
                mean = st.advantages.mean(dim=(0, 1), keepdim=True)
                std = st.advantages.std(dim=(0, 1), keepdim=True) + 1e-8
                st.advantages = (st.advantages - mean) / std
            else:
                st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def _normalize_advantages_batch(self, advantages_batch: torch.Tensor) -> torch.Tensor:
        if self.normalize_advantage_per_critic:
            mean = advantages_batch.mean(dim=0, keepdim=True)
            std = advantages_batch.std(dim=0, keepdim=True) + 1e-8
            return (advantages_batch - mean) / std
        return (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

    def _weighted_actor_advantage(self, advantages_batch: torch.Tensor) -> torch.Tensor:
        return (advantages_batch * self.reward_group_weights).sum(dim=-1)

    def _value_loss(self, value_batch: torch.Tensor, target_values_batch: torch.Tensor, returns_batch: torch.Tensor):
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss_each = torch.max(value_losses, value_losses_clipped).mean(dim=0)
        else:
            value_loss_each = (returns_batch - value_batch).pow(2).mean(dim=0)
        value_loss = (value_loss_each * self.value_loss_weights).sum()
        return value_loss, value_loss_each

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_value_loss_each = torch.zeros(self.num_critics, device=self.device)
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = self._normalize_advantages_batch(advantages_batch)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            adv_total = self._weighted_actor_advantage(advantages_batch)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -adv_total * ratio
            surrogate_clipped = -adv_total * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            value_loss, value_loss_each = self._value_loss(value_batch, target_values_batch, returns_batch)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = torch.nn.MSELoss()(
                    mean_actions_batch[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._clamp_policy_std()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_value_loss_each += value_loss_each.detach()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_value_loss_each /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        for i, name in enumerate(self.critic_names):
            loss_dict[f"value/{name}"] = mean_value_loss_each[i].item()
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict
