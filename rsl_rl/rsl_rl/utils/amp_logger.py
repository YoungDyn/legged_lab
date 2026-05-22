from __future__ import annotations

import git
import os
import pathlib
import statistics
import time
import torch
from collections import deque

import rsl_rl

from rsl_rl.utils.logger import Logger


class LoggerAMP(Logger):
    """Logger class for AMP runners and algorithms."""

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
        max_episode_length_s: float,
    ) -> None:
        super().__init__(
            log_dir,
            cfg,
            env_cfg,
            num_envs,
            is_distributed,
            gpu_world_size,
            gpu_global_rank,
            device,
        )

        # Create buffers for logging AMP rewards and other info
        self.total_rewbuffer = deque(maxlen=100)
        self.style_rewbuffer = deque(maxlen=100)
        self.cur_total_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_style_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reward_group_names = list(self.cfg["algorithm"].get("critic_names", []))
        self.group_rewbuffers = {name: deque(maxlen=100) for name in self.reward_group_names}
        self.raw_group_rewbuffers = {name: deque(maxlen=100) for name in self.reward_group_names}
        self.cur_group_reward_sum = None
        self.cur_raw_group_reward_sum = None
        if self.reward_group_names:
            num_groups = len(self.reward_group_names)
            self.cur_group_reward_sum = torch.zeros(self.num_envs, num_groups, dtype=torch.float, device=self.device)
            self.cur_raw_group_reward_sum = torch.zeros(self.num_envs, num_groups, dtype=torch.float, device=self.device)

        self.max_episode_length_s = max_episode_length_s

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
        style_rewards: torch.Tensor | None = None,
        total_rewards: torch.Tensor | None = None,
    ) -> None:
        """Add metrics from the environment step to the buffers."""
        if self.log_dir is not None:
            if "episode" in extras:
                self.ep_extras.append(extras["episode"])
            elif "log" in extras:
                self.ep_extras.append(extras["log"])

            # Update rewards and episode length
            if intrinsic_rewards is not None:
                self.cur_ereward_sum += rewards
                self.cur_ireward_sum += intrinsic_rewards
                self.cur_reward_sum += rewards + intrinsic_rewards
            else:
                self.cur_reward_sum += rewards
            if style_rewards is not None:
                self.cur_style_reward_sum += style_rewards
            if total_rewards is not None:
                self.cur_total_reward_sum += total_rewards
            reward_groups = extras.get("reward_groups", None)
            raw_reward_groups = extras.get("reward_groups_raw", None)
            reward_group_names = extras.get("reward_group_names", self.reward_group_names)
            if reward_groups is not None and reward_group_names:
                if list(reward_group_names) != self.reward_group_names:
                    self.reward_group_names = list(reward_group_names)
                    self.group_rewbuffers = {name: deque(maxlen=100) for name in self.reward_group_names}
                    self.raw_group_rewbuffers = {name: deque(maxlen=100) for name in self.reward_group_names}
                    self.cur_group_reward_sum = torch.zeros(
                        self.num_envs, len(self.reward_group_names), dtype=torch.float, device=self.device
                    )
                    self.cur_raw_group_reward_sum = torch.zeros(
                        self.num_envs, len(self.reward_group_names), dtype=torch.float, device=self.device
                    )
                self.cur_group_reward_sum += reward_groups.to(self.device)
                if raw_reward_groups is not None:
                    self.cur_raw_group_reward_sum += raw_reward_groups.to(self.device)
            self.cur_episode_length += 1

            # Clear data for completed episodes
            new_ids = (dones > 0).nonzero(as_tuple=False)
            done_env_ids = new_ids.view(-1)
            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            if reward_groups is not None and len(done_env_ids) > 0 and self.cur_group_reward_sum is not None:
                for group_id, group_name in enumerate(self.reward_group_names):
                    self.group_rewbuffers[group_name].extend(
                        self.cur_group_reward_sum[done_env_ids, group_id].cpu().numpy().tolist()
                    )
                    self.raw_group_rewbuffers[group_name].extend(
                        self.cur_raw_group_reward_sum[done_env_ids, group_id].cpu().numpy().tolist()
                    )
                self.cur_group_reward_sum[done_env_ids] = 0
                self.cur_raw_group_reward_sum[done_env_ids] = 0
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0
            if intrinsic_rewards is not None:
                self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.cur_ereward_sum[new_ids] = 0
                self.cur_ireward_sum[new_ids] = 0
            if style_rewards is not None and total_rewards is not None:
                amp_new_ids = new_ids if len(new_ids)>0 else slice(None)
                style_rew_episode_mean = torch.mean(self.cur_style_reward_sum[amp_new_ids]) / (self.max_episode_length_s)
                if len(new_ids) > 0:
                    self.ep_extras[-1]["Episode_Reward/style"] = style_rew_episode_mean.item()
                self.total_rewbuffer.extend(self.cur_total_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.style_rewbuffer.extend(self.cur_style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.cur_style_reward_sum[new_ids] = 0
                self.cur_total_reward_sum[new_ids] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log the training metrics to the logging service and print them to the console."""
        if self.log_dir is not None and not self.disable_logs:
            collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
            iteration_time = collect_time + learn_time
            self.tot_timesteps += collection_size
            self.tot_time += iteration_time

            # Log episode extras
            extras_metrics = []
            if self.ep_extras:
                # Iterate over all keys in the episode info dictionary
                for key in self.ep_extras[0]:
                    infotensor = torch.tensor([], device=self.device)
                    # Iterate over all steps
                    for ep_info in self.ep_extras:
                        # Handle missing, scalar, and zero dimensional tensors
                        if key not in ep_info:
                            continue
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    if "/" in key:
                        self.writer.add_scalar(key, value, it)
                        extras_metrics.append((key, value))
                    else:
                        self.writer.add_scalar("Episode/" + key, value, it)
                        extras_metrics.append((key, value))
            extras_string = self._format_metric_lines(extras_metrics, pad) if extras_metrics else ""

            # Log losses
            for key, value in loss_dict.items():
                self.writer.add_scalar(f"Loss/{key}", value, it)
            self.writer.add_scalar("Loss/learning_rate", learning_rate, it)

            # Log noise std
            self.writer.add_scalar("Policy/mean_noise_std", action_std.mean().item(), it)

            # Log performance
            fps = int(collection_size / (collect_time + learn_time))
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collect_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)

            # Log rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                    self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                    self.writer.add_scalar("Rnd/weight", rnd_weight, it)
                self.writer.add_scalar("AMP/mean_total_reward", statistics.mean(self.total_rewbuffer), it)
                self.writer.add_scalar("AMP/mean_style_reward", statistics.mean(self.style_rewbuffer), it)
                for group_name in self.reward_group_names:
                    if len(self.group_rewbuffers[group_name]) > 0:
                        self.writer.add_scalar(
                            f"RewardGroup/mean_{group_name}_reward",
                            statistics.mean(self.group_rewbuffers[group_name]),
                            it,
                        )
                        self.writer.add_scalar(
                            f"RewardGroupRaw/mean_{group_name}_reward",
                            statistics.mean(self.raw_group_rewbuffers[group_name]),
                            it,
                        )
                self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
                if self.logger_type != "wandb":
                    self.writer.add_scalar(
                        "Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time)
                    )
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
                    )

            log_interval = max(1, int(self.cfg.get("log_interval", 1)))
            done_it = it + 1 - start_it
            should_print = done_it == 1 or done_it % log_interval == 0 or it + 1 == total_it
            if not should_print:
                self.ep_extras.clear()
                return

            # Print to console
            log_string = f"""{"#" * width}\n"""
            log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

            # Print run name if provided
            run_name = self.cfg.get("run_name")
            log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""

            # Print performance
            log_string += (
                f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
                f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
                f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
                f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
            )

            # Print losses
            log_string += self._format_loss_lines(loss_dict, pad)

            # Print rewards and episode length
            if len(self.rewbuffer) > 0:
                reward_metrics = []
                if self.cfg["algorithm"]["rnd_cfg"]:
                    reward_metrics.append(("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer)))
                    reward_metrics.append(("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer)))
                reward_metrics.append(("AMP/mean_total_reward", statistics.mean(self.total_rewbuffer)))
                reward_metrics.append(("AMP/mean_style_reward", statistics.mean(self.style_rewbuffer)))
                for group_name in self.reward_group_names:
                    if len(self.group_rewbuffers[group_name]) > 0:
                        reward_metrics.append(
                            (f"RewardGroup/{group_name}", statistics.mean(self.group_rewbuffers[group_name]))
                        )
                        reward_metrics.append(
                            (f"RewardGroupRaw/{group_name}", statistics.mean(self.raw_group_rewbuffers[group_name]))
                        )
                reward_metrics.append(("Train/mean_reward", statistics.mean(self.rewbuffer)))
                reward_metrics.append(("Train/mean_episode_length", statistics.mean(self.lenbuffer)))
                log_string += self._format_metric_lines(reward_metrics, pad)

            # Print noise std
            log_string += f"""{"Mean action noise std:":>{pad}} {action_std.mean().item():.2f}\n"""

            # Print episode extras
            if not print_minimal:
                log_string += extras_string

            # Print footer
            remaining_it = total_it - start_it - done_it
            eta = self.tot_time / done_it * remaining_it
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
                f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            )
            print(log_string)

            # Clear extras buffer
            self.ep_extras.clear()
