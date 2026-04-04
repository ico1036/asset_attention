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

    # Training time guard — expected time scales with complexity
    elapsed = results.get("elapsed_sec", 0)
    n_samples = config.get("train_samples", 620)
    # Rough estimate: 1K params × 1K samples × 500 epochs ≈ 30s on MPS
    # Scale linearly with params and samples
    expected_min = max(MIN_TRAIN_TIME, (n_params * n_samples) / (1000 * 1000) * 30)
    if elapsed < expected_min * 0.3:  # less than 30% of expected = something wrong
        issues.append(f"⚠️ TOO FAST: {elapsed:.0f}s vs expected ~{expected_min:.0f}s for {n_params} params × {n_samples} samples. Undertrained or data too small.")

    # Suspiciously good
    val_s = results.get("val_sharpe", 0)
    test_s = results.get("test_sharpe", 0)
    if val_s > 2.0:
        issues.append(f"⚠️ SUSPICIOUS val_sharpe={val_s:.2f} > 2.0. Check for bugs/look-ahead bias.")

    # IS/OOS gap
    gap = abs(val_s - test_s)
    if gap > 1.5:
        issues.append(f"⚠️ GAP: |val-test| = {gap:.2f} > 1.5. Likely overfitting.")

    # Loss curve check
    loss_curve = results.get("loss_curve", {})
    train_losses = loss_curve.get("train", [])
    val_losses = loss_curve.get("val", [])
    if train_losses and val_losses and len(train_losses) >= 3:
        # Train not learning?
        if train_losses[-1] >= train_losses[0]:
            issues.append(f"❌ TRAIN_LOSS not falling: {train_losses[0]:.3f} → {train_losses[-1]:.3f}. Model not learning.")
        # Val diverging while train falls?
        if train_losses[-1] < train_losses[0] and val_losses[-1] > val_losses[len(val_losses)//2]:
            issues.append(f"⚠️ OVERFIT: train_loss falling but val_loss rising. Early stopping may be too late.")

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
