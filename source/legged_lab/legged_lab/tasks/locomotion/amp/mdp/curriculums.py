from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def force_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    reward_term_name: str
):
    force_command = env.command_manager.get_term("force_command")
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.6 * reward_term_cfg.weight:
        force_command._command[env_ids, 0] = (force_command._command[env_ids, 0] - 1.0).clamp(min=0.0)
    return torch.mean(torch.squeeze(force_command.command))
