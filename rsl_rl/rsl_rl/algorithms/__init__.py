# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

from .distillation import Distillation
from .dwaq_ppo import DWAQPPO
from .multi_critic_ppo import MultiCriticPPO
from .multi_critic_ppo_amp import MultiCriticPPOAMP
from .ppo import PPO
from .ppo_amp import PPOAMP

__all__ = ["PPO", "DWAQPPO", "Distillation", "PPOAMP", "MultiCriticPPO", "MultiCriticPPOAMP"]
