"""Play G1 walk/run motion pickle files in Isaac Sim and record videos.

This script reuses the Legged Lab animation environment.  It loads the motion
files from ``MotionData/g1_29dof/amp/walk_and_run``, assigns one clip to each
environment in a batch, plays from the first frame, and records the viewport
with Gymnasium's ``RecordVideo`` wrapper.
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import os
import sys
from pathlib import Path

EXTENSION_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = EXTENSION_DIR / "source" / "legged_lab"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record G1 AMP walk/run motion pickle playback videos.")
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help=(
        "Directory containing Legged Lab motion .pkl files. Defaults to "
        "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run."
    ),
)
parser.add_argument(
    "--motions",
    nargs="*",
    default=None,
    help=(
        "Optional motion names, .pkl file paths, or shell-style patterns to record. "
        "Examples: C3_-_run_stageii C*_stageii Walk_*. If omitted, all pkl files are used."
    ),
)
parser.add_argument("--batch_size", type=int, default=9, help="Number of motion clips shown in each recorded video.")
parser.add_argument("--env_spacing", type=float, default=3.0, help="Spacing between parallel playback environments.")
parser.add_argument(
    "--video_dir",
    type=str,
    default="logs/motion_playback/walk_and_run",
    help="Output directory for recorded mp4 files and batch manifests.",
)
parser.add_argument("--video_length", type=int, default=None, help="Override video length in environment steps.")
parser.add_argument("--extra_steps", type=int, default=10, help="Extra hold steps appended after the longest clip.")
parser.add_argument("--fps", type=int, default=None, help="Recorded video FPS. Defaults to the env control rate.")
parser.add_argument("--width", type=int, default=1280, help="Video width in pixels.")
parser.add_argument("--height", type=int, default=720, help="Video height in pixels.")
parser.add_argument("--camera_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--camera_lookat", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument(
    "--follow_robot",
    action="store_true",
    help="Follow robot_anim root with the camera. Intended for --batch_size 1 recordings.",
)
parser.add_argument("--show_keypoints", action="store_true", help="Show red key-body markers during playback.")
parser.add_argument("--no_video", action="store_true", help="Play without writing mp4 files.")
parser.add_argument("--list", action="store_true", help="List discovered motion names and exit.")
parser.add_argument("--robot", type=str, default="g1_29dof", choices=["g1_29dof"], help="Robot model to use.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.no_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch

from isaaclab.utils.dict import print_dict

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.envs import ManagerBasedAnimationEnv
from legged_lab.tasks.locomotion.animation.config.g1.g1_anim_env_cfg import G1AnimEnvCfg


def _default_motion_dir() -> Path:
    return Path(LEGGED_LAB_ROOT_DIR) / "data" / "MotionData" / "g1_29dof" / "amp" / "walk_and_run"


def _discover_motion_names(motion_dir: Path) -> list[str]:
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion directory does not exist: {motion_dir}")
    names = sorted(path.stem for path in motion_dir.glob("*.pkl"))
    if not names:
        raise FileNotFoundError(f"No .pkl motion files found in: {motion_dir}")
    return names


def _select_motion_names(all_names: list[str], selectors: list[str] | None) -> list[str]:
    if not selectors:
        return all_names

    selected: list[str] = []
    for selector in selectors:
        stem = Path(selector).stem if selector.endswith(".pkl") or os.path.sep in selector else selector
        matches = [name for name in all_names if name == stem]
        if not matches:
            matches = [name for name in all_names if fnmatch.fnmatch(name, stem)]
        if not matches:
            raise ValueError(f"No motion matched selector '{selector}'.")
        for name in matches:
            if name not in selected:
                selected.append(name)
    return selected


def _iter_batches(items: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {batch_size}.")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _camera_defaults(num_envs: int, env_spacing: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if num_envs <= 1:
        return (3.0, -3.0, 2.0), (0.0, 0.0, 0.8)

    cols = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / cols)
    span = max(cols, rows) * env_spacing
    eye = (0.55 * span, -0.95 * span, 0.55 * span + 3.0)
    lookat = (0.0, 0.0, 0.8)
    return eye, lookat


def _make_env_cfg(motion_dir: Path, motion_names: list[str]):
    env_cfg = G1AnimEnvCfg()
    env_cfg.scene.num_envs = len(motion_names)
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 3600.0

    env_cfg.viewer.resolution = (args_cli.width, args_cli.height)
    eye, lookat = _camera_defaults(len(motion_names), args_cli.env_spacing)
    env_cfg.viewer.eye = tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else eye
    env_cfg.viewer.lookat = tuple(args_cli.camera_lookat) if args_cli.camera_lookat is not None else lookat
    if args_cli.follow_robot:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot_anim"
        env_cfg.viewer.env_index = 0

    env_cfg.motion_data.motion_dataset.motion_data_dir = str(motion_dir)
    env_cfg.motion_data.motion_dataset.motion_data_weights = {name: 1.0 for name in motion_names}

    env_cfg.animation.animation.random_initialize = False
    env_cfg.animation.animation.random_fetch = False
    env_cfg.animation.animation.num_steps_to_use = 1
    env_cfg.animation.animation.enable_visualization = True
    env_cfg.animation.animation.motion_data_components = ["root_pos_w", "root_quat", "dof_pos", "key_body_pos_b"]

    # Keep finished clips on their last frame instead of resetting into a random new clip.
    env_cfg.terminations.motion_data_finish = None
    return env_cfg


def _assign_motion_ids(env: ManagerBasedAnimationEnv, num_motions: int) -> torch.Tensor:
    animation = env.animation_manager.get_term("animation")
    motion_ids = torch.arange(num_motions, dtype=torch.long, device=env.device)
    env_ids = torch.arange(num_motions, dtype=torch.long, device=env.device)

    animation.motion_ids[env_ids] = motion_ids
    animation.motion_durations[env_ids] = animation.motion_data_term.get_motion_durations(motion_ids)
    animation.motion_fetch_time[env_ids, :] = 0.0
    animation._fetch_motion_data(env_ids)
    animation.key_body_marker.set_visibility(args_cli.show_keypoints)
    animation._visualize()
    return animation.motion_durations[env_ids]


def _write_manifest(video_dir: Path, batch_index: int, motion_names: list[str], video_steps: int, fps: int) -> None:
    manifest_path = video_dir / f"walk_and_run_batch_{batch_index:03d}.txt"
    with manifest_path.open("w", encoding="utf-8") as file:
        file.write(f"video: walk_and_run_batch_{batch_index:03d}-step-0.mp4\n")
        file.write(f"video_steps: {video_steps}\n")
        file.write(f"fps: {fps}\n")
        file.write("motions:\n")
        for env_id, motion_name in enumerate(motion_names):
            file.write(f"  env_{env_id}: {motion_name}\n")


def _record_batch(batch_index: int, motion_dir: Path, motion_names: list[str]) -> None:
    omni.usd.get_context().new_stage()

    env_cfg = _make_env_cfg(motion_dir, motion_names)
    render_mode = None if args_cli.no_video else "rgb_array"
    env = ManagerBasedAnimationEnv(cfg=env_cfg, render_mode=render_mode)

    fps = args_cli.fps if args_cli.fps is not None else round(1.0 / env.step_dt)
    video_dir = Path(args_cli.video_dir).expanduser().resolve()
    if not args_cli.no_video:
        video_kwargs = {
            "video_folder": str(video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length if args_cli.video_length is not None else 1,
            "name_prefix": f"walk_and_run_batch_{batch_index:03d}",
            "fps": fps,
            "disable_logger": True,
        }
        print("[INFO] Recording motion playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env.reset()
    raw_env: ManagerBasedAnimationEnv = env.unwrapped
    motion_durations = _assign_motion_ids(raw_env, len(motion_names))

    auto_video_steps = int(torch.ceil(torch.max(motion_durations) / raw_env.step_dt).item()) + args_cli.extra_steps
    video_steps = args_cli.video_length if args_cli.video_length is not None else auto_video_steps
    if not args_cli.no_video:
        env.video_length = video_steps
        video_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(video_dir, batch_index, motion_names, video_steps, fps)

    action = torch.zeros(raw_env.action_space.shape, device=raw_env.device)
    with torch.inference_mode():
        for _ in range(video_steps + 1):
            if not simulation_app.is_running():
                break
            env.step(action)

    env.close()
    print(
        f"[INFO] Finished batch {batch_index:03d}: {len(motion_names)} motions, "
        f"{video_steps} env steps, output_dir={video_dir}"
    )


def main() -> None:
    motion_dir = Path(args_cli.motion_dir).expanduser().resolve() if args_cli.motion_dir else _default_motion_dir()
    all_motion_names = _discover_motion_names(motion_dir)
    motion_names = _select_motion_names(all_motion_names, args_cli.motions)

    if args_cli.list:
        print(f"[INFO] Motion directory: {motion_dir}")
        for name in motion_names:
            print(name)
        return

    batches = _iter_batches(motion_names, args_cli.batch_size)
    print(f"[INFO] Recording {len(motion_names)} motions from {motion_dir} in {len(batches)} batch(es).")
    for batch_index, batch in enumerate(batches):
        _record_batch(batch_index, motion_dir, batch)


if __name__ == "__main__":
    main()
    simulation_app.close()
