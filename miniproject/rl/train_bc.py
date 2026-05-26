"""Behavior cloning on the CNN policy: train an SB3 MultiInputPolicy to mimic
the heuristic's action choices, given the same Dict observation PPO will use.

Unlike the old flat-MLP BC, this trains the *actual* SB3 policy network
(CombinedExtractor: NatureCNN on the 6-channel vision + a small MLP on the 8
scalars) and saves it as a full PPO `.zip`. Warm-starting PPO is then a plain
`PPO.load()` — no fragile state_dict mapping.

BC trains only the policy path: the shared features extractor + the policy
MLP + the action head. The value head and log_std are left random for PPO to
learn during fine-tuning (BC cannot teach a value function anyway). This is
the "evolved scaffold + lifetime learning" architecture.

Run from cobar-2026/:
    .venv/Scripts/python.exe miniproject/rl/train_bc.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
MP_ROOT = THIS_DIR.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from stable_baselines3 import PPO

from rl.cobar_env import CobarEnv
from rl.policy_config import POLICY_KWARGS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=str, default="miniproject/rl/demos_vision.npz")
    p.add_argument("--out", type=str, default="miniproject/rl/bc_policy.zip")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of demos held out for validation.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- load demos -----
    print(f"Loading demos from {args.demos}")
    data = np.load(args.demos, allow_pickle=True)
    vision = data["vision"]                         # (N, 6, 64, 64) uint8
    scalars = data["scalars"].astype(np.float32)    # (N, 8)
    actions = data["actions"].astype(np.float32)    # (N, 2)
    N = len(vision)
    print(f"  {N:,} transitions")
    print(f"  vision {vision.shape} {vision.dtype}  "
          f"scalars {scalars.shape}  actions {actions.shape}")

    # ----- train/val split (index numpy first, then tensorise the splits) -----
    perm = np.random.permutation(N)
    n_val = int(N * args.val_frac)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    print(f"  train: {len(tr_idx):,}  val: {len(val_idx):,}")

    # vision stays uint8 — SB3's preprocess_obs normalises it /255 internally.
    vision_tr = torch.from_numpy(vision[tr_idx])
    vision_val = torch.from_numpy(vision[val_idx])
    scalars_tr = torch.from_numpy(scalars[tr_idx])
    scalars_val = torch.from_numpy(scalars[val_idx])
    actions_tr = torch.from_numpy(actions[tr_idx])
    actions_val = torch.from_numpy(actions[val_idx])
    del data, vision, scalars, actions

    train_loader = DataLoader(
        TensorDataset(vision_tr, scalars_tr, actions_tr),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(vision_val, scalars_val, actions_val),
        batch_size=args.batch_size,
    )

    # ----- build the SB3 policy we will BC-train -----
    # CobarEnv.__init__ only sets up the spaces (the MuJoCo sim is created
    # lazily in reset()), so this is cheap — no simulation spins up here.
    env = CobarEnv(level=0)
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=POLICY_KWARGS,
        device="auto",
        seed=args.seed,
        verbose=0,
    )
    policy = model.policy
    device = policy.device
    print(f"  policy device: {device}")

    # BC trains the policy path only.
    bc_params = (
        list(policy.features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(bc_params, lr=args.lr)
    loss_fn = nn.MSELoss()

    def forward_mean(vision_b, scalars_b):
        """Dict obs -> action mean, through the SB3 policy network."""
        obs = {"vision": vision_b.to(device), "scalars": scalars_b.to(device)}
        features = policy.extract_features(obs)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        return policy.action_net(latent_pi)

    # ----- BC training loop -----
    best_val = float("inf")
    t_start = time.perf_counter()
    print()
    for epoch in range(args.epochs):
        policy.train()
        tr_loss = 0.0
        n_seen = 0
        for vb, sb, ab in train_loader:
            optimizer.zero_grad()
            pred = forward_mean(vb, sb)
            loss = loss_fn(pred, ab.to(device))
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * vb.size(0)
            n_seen += vb.size(0)
        tr_loss /= max(n_seen, 1)

        policy.eval()
        with torch.no_grad():
            v_loss = 0.0
            n_v = 0
            for vb, sb, ab in val_loader:
                pred = forward_mean(vb, sb)
                v_loss += loss_fn(pred, ab.to(device)).item() * vb.size(0)
                n_v += vb.size(0)
            v_loss /= max(n_v, 1)

        msg = ""
        if v_loss < best_val:
            best_val = v_loss
            model.save(args.out)
            msg = "  *** saved (new best)"
        if epoch % 5 == 0 or epoch == args.epochs - 1 or msg:
            print(f"epoch {epoch:3d}  train_mse={tr_loss:.5f}  "
                  f"val_mse={v_loss:.5f}{msg}")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"BC training done in {elapsed:.0f}s")
    print(f"Best val MSE: {best_val:.5f}")
    print(f"Saved PPO .zip (best val) to: {args.out}")


if __name__ == "__main__":
    main()
