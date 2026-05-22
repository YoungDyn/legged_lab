"""Export a trained DWAQ policy to TorchScript (.pt) for real-robot deployment.

The exported model accepts **two** flat tensors:

1. ``obs_history_flat``  – ``[batch, total_history_dim]`` – the term-major
   observation history produced by Isaac Lab's ``ObservationManager`` (each
   term's full history is flattened, then all terms are concatenated).
2. ``current_obs``       – ``[batch, num_policy_obs]`` – the current-frame
   policy observation (no history).

Internally the model:

1. Normalises ``current_obs``.
2. Encodes ``obs_history_flat`` through the β-VAE encoder → latent code.
3. Feeds ``[code, current_obs]`` to the actor MLP → actions.

Usage
-----
::

    python scripts/rsl_rl/export_dwaq_policy.py \\
        --checkpoint logs/rsl_rl/g1_dwaq/2026-04-09_12-00-00/model_5000.pt

The exported TorchScript file will be saved next to the checkpoint under
``exported/dwaq_policy.pt``.
"""

from __future__ import annotations

import argparse
import copy
import os

import torch
import torch.nn as nn


class DWAQPolicyExporter(nn.Module):
    """Wraps VAE encoder + actor into a single JIT-traceable module.

    Parameters
    ----------
    actor : nn.Module
        The actor MLP (input: ``[code, obs]``; output: actions).
    context_vae : nn.Module
        The ``ContextVAE`` module from ``ActorCriticDWAQ``.
    normalizer : nn.Module | None
        The ``actor_obs_normalizer``.  If ``None``, an ``nn.Identity`` is used.
    num_policy_obs : int
        Dimensionality of one frame of policy observations.
    """

    def __init__(
        self,
        actor: nn.Module,
        context_vae: nn.Module,
        normalizer: nn.Module | None,
        num_policy_obs: int,
    ) -> None:
        super().__init__()
        self.actor = copy.deepcopy(actor)
        # Only keep the encoder path of the VAE (no decoder needed)
        self.encoder = copy.deepcopy(context_vae.encoder)
        self.mean_vel = copy.deepcopy(context_vae.mean_vel)
        self.mean_latent = copy.deepcopy(context_vae.mean_latent)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()
        self.num_policy_obs = num_policy_obs

    def forward(
        self,
        obs_history_flat: torch.Tensor,
        current_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for deployment.

        Args:
            obs_history_flat: ``[batch, total_history_dim]`` – term-major
                flattened observation history (each term's history block is
                contiguous, then all blocks are concatenated).
            current_obs: ``[batch, num_policy_obs]`` – current-frame policy
                observation (no history, same terms but only the latest frame).

        Returns:
            actions: ``[batch, num_actions]``
        """
        current_obs = self.normalizer(current_obs)

        h = self.encoder(obs_history_flat)
        mean_vel = self.mean_vel(h)
        mean_latent = self.mean_latent(h)
        code = torch.cat([mean_vel, mean_latent], dim=-1)

        return self.actor(torch.cat([code, current_obs], dim=-1))


def export_dwaq_policy(checkpoint_path: str, output_path: str | None = None) -> str:
    """Load a DWAQ checkpoint and export the policy as TorchScript.

    Returns the path to the saved ``.pt`` file.
    """
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_sd = ckpt.get("model_state_dict", ckpt)

    # ------------------------------------------------------------------
    # Infer architecture dimensions from the saved state-dict.
    # ------------------------------------------------------------------
    # actor.0.weight → [hidden, actor_input_dim]  where actor_input_dim = code_dim + num_obs
    actor_input_dim = model_sd["actor.0.weight"].shape[1]
    num_actions = [v for k, v in model_sd.items() if k.startswith("actor.") and k.endswith(".bias")][-1].shape[0]

    # context_vae.mean_vel.weight → [vel_dim, encoder_latent]
    vel_dim = model_sd["context_vae.mean_vel.weight"].shape[0]
    latent_dim = model_sd["context_vae.mean_latent.weight"].shape[0]
    cenet_out_dim = vel_dim + latent_dim
    num_policy_obs = actor_input_dim - cenet_out_dim

    encoder_input_dim = model_sd["context_vae.encoder.0.weight"].shape[1]
    obs_history_length = encoder_input_dim // num_policy_obs

    print(f"[INFO] Detected: num_obs={num_policy_obs}, num_actions={num_actions}, "
          f"history={obs_history_length}, code_dim={cenet_out_dim} "
          f"(vel={vel_dim}+latent={latent_dim})")

    # ------------------------------------------------------------------
    # Reconstruct sub-modules from state-dict keys.
    # ------------------------------------------------------------------
    def _build_sequential(prefix: str) -> nn.Sequential:
        """Reconstruct an ``nn.Sequential`` from matching state-dict keys."""
        layers: list[nn.Module] = []
        idx = 0
        while f"{prefix}.{idx}.weight" in model_sd:
            w = model_sd[f"{prefix}.{idx}.weight"]
            b = model_sd[f"{prefix}.{idx}.bias"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            idx += 1
            # Check for activation (no weight → skip index)
            if f"{prefix}.{idx}.weight" not in model_sd and f"{prefix}.{idx + 1}.weight" in model_sd:
                layers.append(nn.ELU())
                idx += 1
            elif idx > 0 and f"{prefix}.{idx}.weight" not in model_sd:
                break
        return nn.Sequential(*layers)

    def _build_linear(prefix: str) -> nn.Linear:
        w = model_sd[f"{prefix}.weight"]
        b = model_sd[f"{prefix}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        return lin

    # Build actor MLP
    actor = _build_sequential("actor")
    # Patch: add ELU activations between linear layers in actor
    actor_layers: list[nn.Module] = []
    linear_keys = sorted([k for k in model_sd if k.startswith("actor.") and k.endswith(".weight")])
    for i, key in enumerate(linear_keys):
        idx_str = key.split(".")[1]
        w = model_sd[f"actor.{idx_str}.weight"]
        b = model_sd[f"actor.{idx_str}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        actor_layers.append(lin)
        if i < len(linear_keys) - 1:
            actor_layers.append(nn.ELU())
    actor = nn.Sequential(*actor_layers)

    # Build encoder
    # NOTE: ContextVAE uses ``last_activation=activation`` so the encoder has
    # an ELU after every linear layer, including the LAST one.
    enc_keys = sorted([k for k in model_sd if k.startswith("context_vae.encoder.") and k.endswith(".weight")])
    enc_layers: list[nn.Module] = []
    for i, key in enumerate(enc_keys):
        idx_str = key.split(".")[2]
        w = model_sd[f"context_vae.encoder.{idx_str}.weight"]
        b = model_sd[f"context_vae.encoder.{idx_str}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        enc_layers.append(lin)
        enc_layers.append(nn.ELU())
    encoder = nn.Sequential(*enc_layers)

    mean_vel = _build_linear("context_vae.mean_vel")
    mean_latent = _build_linear("context_vae.mean_latent")

    # Build normalizer
    normalizer: nn.Module | None = None
    norm_mean_key = "actor_obs_normalizer.running_mean"
    if norm_mean_key in model_sd:
        mean = model_sd[norm_mean_key]
        var = model_sd["actor_obs_normalizer.running_var"]
        count = model_sd.get("actor_obs_normalizer.count", torch.tensor(1.0))

        class _Normalizer(nn.Module):
            def __init__(self, m: torch.Tensor, v: torch.Tensor):
                super().__init__()
                self.register_buffer("mean", m)
                self.register_buffer("var", v)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x - self.mean) / torch.sqrt(self.var + 1e-8)

        normalizer = _Normalizer(mean, var)
        print(f"[INFO] Loaded normalizer (count={count.item():.0f})")

    # ------------------------------------------------------------------
    # Build exporter and trace.
    # ------------------------------------------------------------------
    exporter = DWAQPolicyExporter.__new__(DWAQPolicyExporter)
    nn.Module.__init__(exporter)
    exporter.actor = actor
    exporter.encoder = encoder
    exporter.mean_vel = mean_vel
    exporter.mean_latent = mean_latent
    exporter.normalizer = normalizer if normalizer is not None else nn.Identity()
    exporter.num_policy_obs = num_policy_obs
    exporter.eval()

    test_history = torch.zeros(1, encoder_input_dim)
    test_obs = torch.zeros(1, num_policy_obs)
    test_output = exporter(test_history, test_obs)
    print(f"[INFO] Test — history: {test_history.shape}, obs: {test_obs.shape}, "
          f"output: {test_output.shape}")

    # Trace and save
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "dwaq_policy.pt")

    traced = torch.jit.trace(exporter, (test_history, test_obs))
    traced.save(output_path)
    print(f"[INFO] Exported TorchScript to: {output_path}")

    # Quick verification
    loaded = torch.jit.load(output_path)
    verify_out = loaded(test_history, test_obs)
    assert torch.allclose(test_output, verify_out, atol=1e-6), "Verification failed!"
    print("[INFO] Verification passed.")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DWAQ policy to TorchScript")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_xxxx.pt")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path (default: <checkpoint_dir>/exported/dwaq_policy.pt)")
    args = parser.parse_args()

    export_dwaq_policy(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
