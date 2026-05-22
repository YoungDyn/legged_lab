from __future__ import annotations

import torch

from isaaclab.envs import VecEnvStepReturn
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    RecorderManager,
    RewardManager,
    TerminationManager,
)

from legged_lab.managers import AnimationManager, MotionDataManager, PreviewObservationManager

from .manager_based_amp_env_cfg import ManagerBasedAmpEnvCfg
from .manager_based_animation_env import ManagerBasedAnimationEnv


class ManagerBasedAmpEnv(ManagerBasedAnimationEnv):
    """AMP environment with terminal observation export and optional multi-critic reward grouping."""

    cfg: ManagerBasedAmpEnvCfg

    def __init__(self, cfg: ManagerBasedAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _merge_terminal_obs(
        self,
        current_obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        preview_obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        reset_env_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Merge pre-reset previews into post-reset observations for terminated envs."""
        terminal_obs = {}
        for key, value in current_obs.items():
            if key not in preview_obs:
                terminal_obs[key] = value
                continue
            preview_value = preview_obs[key]
            if isinstance(value, dict) and isinstance(preview_value, dict):
                terminal_obs[key] = self._merge_terminal_obs(value, preview_value, reset_env_ids)
            elif isinstance(value, torch.Tensor) and isinstance(preview_value, torch.Tensor):
                merged_value = value.clone()
                merged_value[reset_env_ids] = preview_value[reset_env_ids]
                terminal_obs[key] = merged_value
            else:
                terminal_obs[key] = value
        return terminal_obs

    def _preview_terminal_obs(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]] | None:
        """Preview only the configured terminal observation groups before reset."""
        group_names = tuple(getattr(self.cfg, "terminal_obs_groups", ("disc",)))
        if not group_names:
            return None

        if hasattr(self.observation_manager, "preview_group"):
            preview_obs = {}
            for group_name in group_names:
                preview_obs[group_name] = self.observation_manager.preview_group(group_name)
            return preview_obs

        if hasattr(self.observation_manager, "preview"):
            preview_obs = self.observation_manager.preview()
            return {group_name: preview_obs[group_name] for group_name in group_names}

        return None

    def _compute_reward_groups(self) -> tuple[torch.Tensor, list[str]] | None:
        """Compute dt-scaled reward groups configured for multi-critic algorithms."""
        if self.cfg.reward_groups is None:
            return None

        critic_names = list(self.cfg.reward_groups.keys())
        reward_groups_tensor = torch.zeros((self.num_envs, len(critic_names)), device=self.device, dtype=torch.float32)
        term_names = list(self.reward_manager._term_names)
        term_name_to_index = {term_name: i for i, term_name in enumerate(term_names)}

        # RewardManager.compute(dt=...) returns dt-scaled totals, while _step_reward stores per-term values before dt.
        step_rewards_with_dt = self.reward_manager._step_reward * self.step_dt

        for critic_index, critic_name in enumerate(critic_names):
            for term_name in self.cfg.reward_groups[critic_name]:
                term_index = term_name_to_index.get(term_name)
                if term_index is None:
                    raise KeyError(
                        f"Reward group '{critic_name}' references unknown reward term '{term_name}'."
                        f" Available terms are: {term_names}"
                    )
                reward_groups_tensor[:, critic_index] += step_rewards_with_dt[:, term_index]

        return reward_groups_tensor, critic_names

    def load_managers(self):
        """Load AMP-specific managers while swapping in the local preview observation manager."""
        self.motion_data_manager = MotionDataManager(self.cfg.motion_data, self)
        print("[INFO] Motion Data Manager: ", self.motion_data_manager)
        self.animation_manager = AnimationManager(self.cfg.animation, self)
        print("[INFO] Animation Manager: ", self.animation_manager)

        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        print("[INFO] Event Manager: ", self.event_manager)
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        self.observation_manager = PreviewObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and expose terminal observations in ``extras``."""
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update animation manager
        self.animation_manager.update(dt=self.step_dt)
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reward_groups = self._compute_reward_groups()
        if reward_groups is not None:
            reward_groups_tensor, critic_names = reward_groups
            self.extras["reward_groups"] = reward_groups_tensor
            self.extras["reward_group_names"] = critic_names
        else:
            self.extras.pop("reward_groups", None)
            self.extras.pop("reward_group_names", None)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        terminal_obs_preview = None
        if len(reset_env_ids) > 0:
            terminal_obs_preview = self._preview_terminal_obs()

        # -- reset envs that terminated/timed-out and log the episode information
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        terminal_obs = None
        if terminal_obs_preview is not None:
            for group_name in terminal_obs_preview:
                if group_name not in self.obs_buf:
                    raise KeyError(
                        f"Configured terminal observation group '{group_name}' is not present in current observations."
                    )
            current_terminal_groups = {group_name: self.obs_buf[group_name] for group_name in terminal_obs_preview}
            terminal_obs = self._merge_terminal_obs(current_terminal_groups, terminal_obs_preview, reset_env_ids)
        if terminal_obs is not None:
            self.extras["terminal_obs"] = terminal_obs
        else:
            self.extras.pop("terminal_obs", None)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
