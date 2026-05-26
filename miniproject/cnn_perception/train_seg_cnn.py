"""Train the segmentation CNN on the privileged-label dataset from
gen_seg_labels.py.

Each eye is an independent training example: (3, H, W) RGB -> (H, W) class
mask. Episodes (level, seed) are split train/val so the val metric reflects
generalisation to unseen grass layouts — not memorised consecutive frames.

Headline metric: grass IoU — that is the class the controller needs.

Run from cobar-2026/:
    .venv/Scripts/python.exe miniproject/cnn_perception/train_seg_cnn.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
MP_ROOT = THIS_DIR.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from cnn_perception.seg_model import SegCNN, save_seg_cnn, load_seg_cnn, N_CLASSES

CLASS_COLORS = np.array([[0, 0, 0], [0, 255, 0], [0, 255, 255]], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                   default="miniproject/cnn_perception/seg_data.npz")
    p.add_argument("--out", type=str,
                   default="miniproject/cnn_perception/seg_cnn.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of EPISODES held out for validation.")
    p.add_argument("--base", type=int, default=32, help="U-Net base width.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def per_class_iou(conf):
    """conf: (C,C) confusion matrix, rows=true, cols=pred -> IoU per class."""
    iou = np.full(conf.shape[0], np.nan)
    for c in range(conf.shape[0]):
        tp = conf[c, c]
        denom = conf[c, :].sum() + conf[:, c].sum() - tp
        if denom > 0:
            iou[c] = tp / denom
    return iou


def save_viz(model, vis_u8, msk, idx, device, out_path, n=6):
    """Save RGB | ground-truth mask | predicted mask panels for n examples."""
    n = min(n, len(idx))
    rows = []
    model.eval()
    with torch.no_grad():
        for i in idx[:n]:
            rgb = vis_u8[i]                                   # (H,W,3) uint8
            x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
            pred = model(x.to(device)).argmax(1)[0].cpu().numpy()
            true_c = CLASS_COLORS[msk[i]]
            pred_c = CLASS_COLORS[pred]
            rows.append(np.hstack([rgb, true_c, pred_c]))
    img = np.vstack(rows)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.data}")
    data = np.load(args.data)
    vision = data["vision"]   # (N, 2, H, W, 3) uint8
    mask = data["mask"]       # (N, 2, H, W) uint8
    meta = data["meta"]       # (N, 3) int (level, seed, step)
    N, _, H, W, _ = vision.shape
    print(f"  {N} frames -> {2 * N} per-eye examples, {H}x{W}")

    # --- per-eye examples ---
    vis = np.ascontiguousarray(vision.reshape(2 * N, H, W, 3))
    msk = np.ascontiguousarray(mask.reshape(2 * N, H, W))
    epi = np.repeat(meta[:, :2], 2, axis=0)            # (2N, 2) = (level, seed)

    # --- episode-level train/val split, stratified by SUBSTANTIAL grass so
    #     the val set always contains a grass-rich episode (grass IoU is the
    #     headline metric; a trace-grass episode makes it unmeasurable) ---
    GRASS_EPISODE_MIN = 0.01  # >1% grass pixels => a genuine grass episode
    episodes = np.unique(epi, axis=0)
    has_grass = np.array([
        float((msk[np.all(epi == ep, axis=1)] == 1).mean()) > GRASS_EPISODE_MIN
        for ep in episodes
    ])
    rng = np.random.default_rng(args.seed)
    val_ep_mask = np.zeros(len(episodes), dtype=bool)
    for grp in (np.where(has_grass)[0], np.where(~has_grass)[0]):
        grp = grp.copy()
        rng.shuffle(grp)
        if len(grp):
            n_val = max(1, int(round(len(grp) * args.val_frac)))
            val_ep_mask[grp[:n_val]] = True
    val_epis = episodes[val_ep_mask]
    is_val = np.any(np.all(epi[:, None, :] == val_epis[None, :, :], axis=2), axis=1)
    tr_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    print(f"  episodes {len(episodes)} ({int(has_grass.sum())} with grass) -> "
          f"train {len(episodes) - len(val_epis)}, val {len(val_epis)}  "
          f"{val_epis.tolist()}")
    print(f"  examples -> train {len(tr_idx)}, val {len(val_idx)}")

    vis_t = torch.from_numpy(vis)              # (2N,H,W,3) uint8
    msk_t = torch.from_numpy(msk).long()       # (2N,H,W)

    tr_loader = DataLoader(TensorDataset(vis_t[tr_idx], msk_t[tr_idx]),
                           batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(vis_t[val_idx], msk_t[val_idx]),
                            batch_size=args.batch_size)

    # --- class weights from train pixel counts (inverse frequency, clipped) ---
    counts = np.bincount(msk[tr_idx].ravel(), minlength=N_CLASSES).astype(np.float64)
    inv = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    weights = np.clip(inv, 0.5, 30.0)
    print(f"  pixel counts {counts.astype(int).tolist()}  "
          f"class weights {np.round(weights, 2).tolist()}")
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    model = SegCNN(in_ch=3, n_classes=N_CLASSES, base=args.base).to(device)
    print(f"  SegCNN params {sum(p.numel() for p in model.parameters()):,}  "
          f"device {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights_t)

    def to_input(vb):
        return vb.to(device).permute(0, 3, 1, 2).float() / 255.0

    best_grass_iou = -1.0
    t_start = time.perf_counter()
    print()
    for epoch in range(args.epochs):
        model.train()
        tr_loss, n_seen = 0.0, 0
        for vb, mb in tr_loader:
            x, y = to_input(vb), mb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        tr_loss /= max(n_seen, 1)

        model.eval()
        conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        v_loss, n_v = 0.0, 0
        with torch.no_grad():
            for vb, mb in val_loader:
                x, y = to_input(vb), mb.to(device)
                logits = model(x)
                v_loss += loss_fn(logits, y).item() * x.size(0)
                n_v += x.size(0)
                t = y.view(-1).cpu().numpy()
                p = logits.argmax(1).view(-1).cpu().numpy()
                conf += np.bincount(t * N_CLASSES + p,
                                    minlength=N_CLASSES ** 2
                                    ).reshape(N_CLASSES, N_CLASSES)
        v_loss /= max(n_v, 1)
        iou = per_class_iou(conf)
        grass_iou = iou[1] if not np.isnan(iou[1]) else -1.0

        msg = ""
        if grass_iou > best_grass_iou:
            best_grass_iou = grass_iou
            save_seg_cnn(model, args.out)
            msg = "  *** saved (best grass IoU)"
        if epoch % 5 == 0 or epoch == args.epochs - 1 or msg:
            print(f"epoch {epoch:3d}  train_loss={tr_loss:.4f}  "
                  f"val_loss={v_loss:.4f}  "
                  f"IoU bg/grass/banana={iou[0]:.3f}/{iou[1]:.3f}/{iou[2]:.3f}{msg}")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"Done in {elapsed:.0f}s. Best val grass IoU: {best_grass_iou:.3f}")
    print(f"Saved to {args.out}")

    # --- visual sanity check on held-out val examples ---
    viz_path = Path(args.out).with_name("seg_cnn_val_preds.png")
    best = load_seg_cnn(args.out, device=device)
    save_viz(best, vis, msk, val_idx, device, viz_path, n=6)
    print(f"Saved val predictions visualization to {viz_path}")
    print("  (each row: RGB | ground-truth mask | predicted mask)")


if __name__ == "__main__":
    main()
