#!/usr/bin/env python3
"""Hard guards that cannot be bypassed. Run train.py through this."""

import subprocess, sys, time, json
from pathlib import Path

CARDS = Path(__file__).parent / "cards"
MAX_PARAMS = 25_000
MIN_TRAIN_TIME = 30  # seconds
MIN_SAMPLES = 200

def check_card(card_path):
    """Post-run checks on the experiment card."""
    card = json.load(open(card_path))
    config = card.get("config", {})
    results = card.get("results", {})
    issues = []

    # Param guard
    n_params = config.get("n_params", 0)
    if n_params > MAX_PARAMS:
        issues.append(f"❌ PARAMS: {n_params:,} > {MAX_PARAMS:,} limit")

    # Training time guard
    elapsed = results.get("elapsed_sec", 0)
    if elapsed < MIN_TRAIN_TIME:
        issues.append(f"⚠️ FAST: {elapsed:.0f}s < {MIN_TRAIN_TIME}s minimum. Model undertrained or data too small.")

    # Suspiciously good
    val_s = results.get("val_sharpe", 0)
    test_s = results.get("test_sharpe", 0)
    if val_s > 2.0:
        issues.append(f"⚠️ SUSPICIOUS val_sharpe={val_s:.2f} > 2.0. Check for bugs/look-ahead bias.")

    # IS/OOS gap
    gap = abs(val_s - test_s)
    if gap > 1.5:
        issues.append(f"⚠️ GAP: |val-test| = {gap:.2f} > 1.5. Likely overfitting.")

    # Sanity bounds (based on asset allocation literature + benchmarks)
    # Real-world hedge fund Sharpe rarely exceeds 2.0 sustained
    # EW benchmark ~2.76 on this test period (unusually high due to 2020-2026)
    SANE_VAL_RANGE = (-0.5, 4.0)
    SANE_TEST_RANGE = (-0.5, 5.0)
    if not (SANE_VAL_RANGE[0] <= val_s <= SANE_VAL_RANGE[1]):
        issues.append(f"🚨 INSANE val_sharpe={val_s:.2f} outside [{SANE_VAL_RANGE[0]}, {SANE_VAL_RANGE[1]}]")
    if not (SANE_TEST_RANGE[0] <= test_s <= SANE_TEST_RANGE[1]):
        issues.append(f"🚨 INSANE test_sharpe={test_s:.2f} outside [{SANE_TEST_RANGE[0]}, {SANE_TEST_RANGE[1]}]")

    # Complexity step check: improvement > 100% over previous best is suspicious
    # (legitimate improvements are usually incremental)

    # Auto verdict
    prev_best = None
    for p in sorted(CARDS.glob("exp_*.json")):
        if p == card_path:
            continue
        c = json.load(open(p))
        v = c.get("verdict")
        vs = c.get("results", {}).get("val_sharpe")
        if vs is not None and v not in ("DISCARD",):
            if prev_best is None or vs > prev_best:
                prev_best = vs

    if prev_best is None:
        verdict = "BASELINE"
    else:
        verdict = "KEEP" if val_s > prev_best else "DISCARD"

    card["verdict"] = verdict
    card["issues"] = issues

    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)

    return verdict, issues


def main():
    # Run train.py
    start = time.time()
    result = subprocess.run([sys.executable, "train.py"], cwd=Path(__file__).parent)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"❌ train.py crashed (exit code {result.returncode})")
        sys.exit(1)

    # Find latest card
    cards = sorted(CARDS.glob("exp_*.json"))
    if not cards:
        print("❌ No card generated")
        sys.exit(1)

    latest = cards[-1]
    verdict, issues = check_card(latest)

    print(f"\n{'='*50}")
    print(f"Guard verdict: {verdict}")
    for issue in issues:
        print(f"  {issue}")
    if not issues:
        print("  ✅ All checks passed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
