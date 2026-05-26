"""Quick analyzer for the diagnostic history JSONs from eval_matrix --dump-history."""
import json
import sys
from pathlib import Path


def analyze(path):
    with open(path) as f:
        data = json.load(f)
    print(f"\n=== {Path(path).name} ===")
    print(f"  level={data['level']} seed={data['seed']} outcome={data['outcome']} "
          f"end_step={data['end_step']} min_dist={data['min_dist']:.2f}")
    hist = data["history"]
    n = len(hist)
    print(f"  history entries: {n}")
    if n == 0:
        return

    # mode distribution
    modes = {}
    for h in hist:
        m = h.get("mode", "?")
        modes[m] = modes.get(m, 0) + 1
    print(f"  mode counts: {modes}")

    # tilt stats
    pitches = [h.get("pitch", 0.0) for h in hist]
    rolls = [h.get("roll", 0.0) for h in hist]
    print(f"  pitch: min={min(pitches):+.1f}  max={max(pitches):+.1f}")
    print(f"  roll : min={min(rolls):+.1f}  max={max(rolls):+.1f}")

    # last 50 steps in detail (most diagnostic)
    print("\n  last 50 steps:")
    fields = ["mode", "mode_step", "pitch", "roll", "drive_L", "drive_R",
              "n_central_blades", "clutter_factor"]
    # add dragon_loom if present
    if hist and "dragon_loom" in hist[0]:
        fields.append("dragon_loom")
    print("  " + "  ".join(f"{f:>10}" for f in fields))
    for h in hist[-50:]:
        cells = []
        for f in fields:
            v = h.get(f, "")
            if isinstance(v, float):
                cells.append(f"{v:>10.2f}")
            elif isinstance(v, bool):
                cells.append(f"{str(v):>10}")
            elif isinstance(v, int):
                cells.append(f"{v:>10d}")
            else:
                cells.append(f"{str(v):>10}")
        print("  " + "  ".join(cells))

    # also print step where dragon_loom first became True (if applicable)
    if hist and "dragon_loom" in hist[0]:
        for i, h in enumerate(hist):
            if h.get("dragon_loom"):
                print(f"\n  first dragon_loom=True at history index {i} "
                      f"(deque holds last 15000; episode end was step {data['end_step']})")
                break

    # threshold first-crossings (offset = absolute step at history index 0)
    end_step = data["end_step"]
    offset = end_step - n
    print("\n  threshold first-crossings:")
    for thr in (25, 30, 40, 50, 60, 80):
        for i, h in enumerate(hist):
            if abs(h.get("pitch", 0.0)) > thr:
                print(f"    |pitch|>{thr:>2}: step ~{i + offset}")
                break
        else:
            print(f"    |pitch|>{thr:>2}: never")
    for thr in (25, 40, 50, 60):
        for i, h in enumerate(hist):
            if abs(h.get("roll", 0.0)) > thr:
                print(f"    |roll|>{thr:>2}: step ~{i + offset}")
                break
        else:
            print(f"    |roll|>{thr:>2}: never")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
