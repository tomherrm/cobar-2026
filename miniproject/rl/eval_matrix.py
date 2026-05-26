"""Multi-condition evaluation matrix for the report's central comparison.

Runs each controller across (level, seed) pairs and reports outcome + minimum
distance to banana — the metric that actually predicts hidden-seed success
(binary scoring at radius 3; sitting still far from the banana scores the same
as flipping).

Conditions:
  random      uniform random drive in [-1.8, 1.8] (floor baseline)
  heuristic   the submitted state-machine controller (submission/controller.py)
  bc          the behaviour-cloned policy             (--bc PATH .zip)
  bc_ppo      BC warm-start + PPO fine-tune            (--bc-ppo PATH .zip)
  ppo_scratch PPO trained from scratch                (--ppo-scratch PATH .zip)

A condition is only run if it is listed in --conditions AND (for the policy
conditions) its .zip path exists.

Run from cobar-2026/:
    .venv/Scripts/python.exe miniproject/rl/eval_matrix.py
    .venv/Scripts/python.exe miniproject/rl/eval_matrix.py --conditions heuristic,bc_ppo --levels 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
MP_ROOT = THIS_DIR.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from miniproject.simulation import MiniprojectSimulation
from flygym.compose import ActuatorType
from flygym.examples.locomotion import TurningController
from rl.cobar_env import _build_observation
from submission import movement_correction

SMOOTH_ALPHA = 0.7        # matches CobarEnv / training / deployment
SUCCESS_RADIUS = 3.0      # project scoring radius


# --------------------------------------------------------------------------
# Episode runners
# --------------------------------------------------------------------------

def run_policy_episode(predict_fn, level: int, seed: int, max_steps: int):
    """Run one episode driven by `predict_fn(obs_dict) -> action (2,)`.

    Used for the random / bc / bc_ppo / ppo_scratch conditions — they all
    share the obs-build + smoothing + step loop, only the action source
    differs. Returns (outcome, steps, min_dist).
    """
    sim = MiniprojectSimulation(level=level, seed=seed)
    turning = TurningController(sim.timestep)
    banana_xy = np.asarray(sim.world.banana_xy, dtype=np.float32)
    last_action = np.zeros(2, dtype=np.float32)
    smoothed = np.zeros(2, dtype=np.float32)
    min_dist = float("inf")

    for step in range(max_steps):
        fly_name = sim.fly.name
        olfaction = sim.get_olfaction(fly_name)
        quat = sim.get_body_rotations(fly_name)[0]
        raw_vision = sim.get_raw_vision(fly_name)
        contact_forces_world = sim.get_external_force(
            fly_name, subtract_adhesion_force=True
        )
        try:
            antenna_data = sim.get_antenna_data(fly_name)
        except Exception:
            antenna_data = None
        fly_xy = np.asarray(sim.get_body_positions(fly_name)[0][:2])
        _, _, pitch, roll = movement_correction.tilt_to_control_signal(
            quat, 10, 50, 0.5, 0.3
        )

        obs = _build_observation(
            olfaction, quat, pitch, roll, raw_vision,
            contact_forces_world, antenna_data, last_action,
        )
        action = np.clip(predict_fn(obs), -1.8, 1.8).astype(np.float32)
        smoothed = (1.0 - SMOOTH_ALPHA) * smoothed + SMOOTH_ALPHA * action
        last_action = smoothed

        joint_angles, adhesion = turning.step(smoothed)
        sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(fly_name, ActuatorType.ADHESION, adhesion)
        sim.step()

        dist = float(np.linalg.norm(fly_xy - banana_xy))
        if dist < min_dist:
            min_dist = dist
        if dist <= SUCCESS_RADIUS:
            return "success", step + 1, min_dist
        if abs(pitch) > 90 or abs(roll) > 85:
            return "flip", step + 1, min_dist

    return "timeout", max_steps, min_dist


def _make_heuristic(sim):
    from submission.controller import Controller
    return Controller(sim)


def _make_cnn(sim, cnn_path):
    from cnn_perception.cnn_controller import CNNController
    return CNNController(sim, cnn_path=cnn_path)


def _make_reactive(sim, cnn_path):
    from cnn_perception.reactive_controller import ReactiveController
    return ReactiveController(sim, cnn_path=cnn_path)


def run_controller_episode(make_controller, level: int, seed: int, max_steps: int,
                            history_out: str | None = None):
    """Run one episode with a Controller-style object (heuristic or
    CNNController) — it manages its own state machine and takes the sim
    directly. `make_controller` is a callable: sim -> controller.
    If `history_out` is given, the controller's `_history` deque is written
    there as JSON at end-of-episode (for offline diagnosis).
    Returns (outcome, steps, min_dist).
    """
    sim = MiniprojectSimulation(level=level, seed=seed)
    controller = make_controller(sim)
    banana_xy = np.asarray(sim.world.banana_xy, dtype=np.float32)
    min_dist = float("inf")

    def _dump_history(outcome, end_step):
        if history_out is None:
            return
        try:
            import json
            hist = list(getattr(controller, "_history", []))
            meta = {
                "level": level,
                "seed": seed,
                "outcome": outcome,
                "end_step": int(end_step),
                "min_dist": float(min_dist),
                "history": hist,
            }
            with open(history_out, "w", encoding="utf-8") as f:
                json.dump(meta, f, default=float)
        except Exception as e:
            print(f"  [history dump failed: {e}]")

    for step in range(max_steps):
        fly_name = sim.fly.name
        fly_xy = np.asarray(sim.get_body_positions(fly_name)[0][:2])
        quat = sim.get_body_rotations(fly_name)[0]
        _, _, pitch, roll = movement_correction.tilt_to_control_signal(
            quat, 10, 50, 0.5, 0.3
        )

        joint_angles, adhesion = controller.step(sim)
        sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(fly_name, ActuatorType.ADHESION, adhesion)
        sim.step()

        dist = float(np.linalg.norm(fly_xy - banana_xy))
        if dist < min_dist:
            min_dist = dist
        if dist <= SUCCESS_RADIUS:
            _dump_history("success", step + 1)
            return "success", step + 1, min_dist
        if abs(pitch) > 90 or abs(roll) > 85:
            _dump_history("flip", step + 1)
            return "flip", step + 1, min_dist

    _dump_history("timeout", max_steps)
    return "timeout", max_steps, min_dist


# --------------------------------------------------------------------------
# Condition setup
# --------------------------------------------------------------------------

def build_predict_fns(args):
    """Return {condition_name: predict_fn or None} for the requested,
    available conditions. 'heuristic' maps to None and is handled separately.
    """
    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
    fns = {}

    for name in requested:
        if name == "random":
            rng = np.random.default_rng(0)
            fns["random"] = lambda obs, _rng=rng: _rng.uniform(-1.8, 1.8, size=2)
        elif name == "heuristic":
            fns["heuristic"] = "HEURISTIC"  # sentinel — special-cased in main
        elif name == "cnn":
            if args.cnn is None or not Path(args.cnn).exists():
                print(f"  [skip] cnn: no segmentation CNN at {args.cnn}")
                continue
            fns["cnn"] = "CNN"  # sentinel — special-cased in main
        elif name == "reactive":
            if args.cnn is None or not Path(args.cnn).exists():
                print(f"  [skip] reactive: no segmentation CNN at {args.cnn}")
                continue
            fns["reactive"] = "REACTIVE"  # sentinel — special-cased in main
        elif name in ("bc", "bc_ppo", "ppo_scratch"):
            path = {
                "bc": args.bc,
                "bc_ppo": args.bc_ppo,
                "ppo_scratch": args.ppo_scratch,
            }[name]
            if path is None or not Path(path).exists():
                print(f"  [skip] {name}: no .zip at {path}")
                continue
            from stable_baselines3 import PPO
            model = PPO.load(path, device="auto")
            fns[name] = lambda obs, _m=model: _m.predict(obs, deterministic=True)[0]
        else:
            print(f"  [skip] unknown condition {name!r}")
    return fns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", type=str,
                   default="random,heuristic,bc,bc_ppo,cnn,reactive",
                   help="Comma-separated subset of: random, heuristic, bc, "
                        "bc_ppo, ppo_scratch, cnn, reactive.")
    p.add_argument("--levels", type=str, default="0,1,2")
    p.add_argument("--seeds", type=str, default="1,67,777")
    p.add_argument("--max-steps", type=int, default=100_000,
                   help="Per-episode step cap (project scoring budget).")
    p.add_argument("--bc", type=str, default="miniproject/rl/bc_policy.zip")
    p.add_argument("--bc-ppo", type=str,
                   default="miniproject/rl/checkpoints_bc_ppo_L2_tilt/policy.zip")
    p.add_argument("--ppo-scratch", type=str, default=None)
    p.add_argument("--cnn", type=str,
                   default="miniproject/cnn_perception/seg_cnn.pt",
                   help="Segmentation CNN .pt for the 'cnn' (heuristic+CNN) condition.")
    p.add_argument("--out", type=str, default="miniproject/rl/eval_matrix_results.txt",
                   help="Write the results table here as well as stdout.")
    p.add_argument("--dump-history", action="store_true",
                   help="Dump controller._history per episode as JSON next to --out.")
    return p.parse_args()


def main():
    args = parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print("Loading conditions...")
    fns = build_predict_fns(args)
    if not fns:
        print("No runnable conditions. Nothing to do.")
        return
    print(f"Conditions: {list(fns)}")
    print(f"Levels: {levels}  Seeds: {seeds}  max_steps: {args.max_steps:,}")

    lines = []
    lines.append(f"{'condition':>12} {'L':>2} {'seed':>4}  {'outcome':>8}  "
                 f"{'steps':>7}  {'min_dist':>8}")
    lines.append("-" * 52)

    # success tally: {condition: {level: n_success}}
    tally = {c: {lv: 0 for lv in levels} for c in fns}

    t_start = time.perf_counter()
    for cond, fn in fns.items():
        for level in levels:
            for seed in seeds:
                t0 = time.perf_counter()
                history_out = None
                if args.dump_history:
                    out_root = Path(args.out)
                    history_out = str(
                        out_root.with_name(
                            f"{out_root.stem}_hist_L{level}_s{seed}.json"
                        )
                    )
                if fn == "HEURISTIC":
                    outcome, steps, mind = run_controller_episode(
                        _make_heuristic, level, seed, args.max_steps,
                        history_out=history_out,
                    )
                elif fn == "CNN":
                    outcome, steps, mind = run_controller_episode(
                        lambda s: _make_cnn(s, args.cnn),
                        level, seed, args.max_steps,
                        history_out=history_out,
                    )
                elif fn == "REACTIVE":
                    outcome, steps, mind = run_controller_episode(
                        lambda s: _make_reactive(s, args.cnn),
                        level, seed, args.max_steps,
                        history_out=history_out,
                    )
                else:
                    outcome, steps, mind = run_policy_episode(
                        fn, level, seed, args.max_steps
                    )
                dt = time.perf_counter() - t0
                if outcome == "success":
                    tally[cond][level] += 1
                row = (f"{cond:>12} {level:>2} {seed:>4}  {outcome:>8}  "
                       f"{steps:>7}  {mind:8.2f}")
                print(f"{row}  ({dt:5.0f}s)")
                lines.append(row)

    # summary
    lines.append("")
    lines.append("=== success counts (out of "
                 f"{len(seeds)} seeds) ===")
    header = f"{'condition':>12}  " + "  ".join(f"L{lv}" for lv in levels)
    lines.append(header)
    for cond in fns:
        cells = "  ".join(f"{tally[cond][lv]:>2}" for lv in levels)
        lines.append(f"{cond:>12}  {cells}")

    elapsed = time.perf_counter() - t_start
    lines.append("")
    lines.append(f"total wall time: {elapsed:.0f}s")

    table = "\n".join(lines)
    print()
    print(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table, encoding="utf-8")
    print()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
