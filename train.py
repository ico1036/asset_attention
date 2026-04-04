#!/usr/bin/env python3
"""
Exp 108: Post-refactor baseline — LW MinVar w500 using prepare.py evaluation harness.
Hypothesis: Results should match pre-refactor but with CORRECT Sharpe annualization.
Expected: val_sharpe [0.5, 3.0], test_sharpe [1.0, 4.0], train_time [1, 30]
(Lower Sharpe expected due to sqrt(50.4) instead of sqrt(252))
"""

import time
import torch
from prepare import (
    load_data, make_samples, split_data,
    lw_cov, minvar_weights, equal_weight,
    evaluate_and_print, write_card,
    N_ASSETS, REBAL_FREQ,
)

def main():
    t0 = time.time()

    # Load data
    d = load_data()
    ret = d["log_return"]  # (T, N)

    # Create rebalancing samples
    sample_indices, Y = make_samples(ret)
    ns = len(Y)
    print(f"Samples: {ns} (rebal every {REBAL_FREQ} days)")

    # Split
    train_end, val_end = split_data(ns)
    print(f"Split: train={train_end}, val={val_end - train_end}, test={ns - val_end}")

    # Compute MinVar weights with 500-day lookback
    WINDOW = 500
    weights = torch.stack([
        minvar_weights(lw_cov(ret[max(0, idx - WINDOW):idx]))
        for idx in sample_indices
    ])

    # Evaluate on each split
    val_results = evaluate_and_print(
        weights[train_end:val_end], Y[train_end:val_end],
        "val", benchmark_n=N_ASSETS
    )
    test_results = evaluate_and_print(
        weights[val_end:], Y[val_end:],
        "test", benchmark_n=N_ASSETS
    )
    full_results = evaluate_and_print(
        weights, Y,
        "full", benchmark_n=N_ASSETS
    )

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.1f}s")

    # Write card
    config = {
        "model": "LW_MinVar_w500",
        "n_params": 0,
        "train_samples": train_end,
        "n_assets": N_ASSETS,
        "window": WINDOW,
    }
    card_results = {
        "val_sharpe": val_results["sharpe"],
        "test_sharpe": test_results["sharpe"],
        "full_sharpe": full_results["sharpe"],
        "val_ew_sharpe": val_results.get("ew_sharpe", 0),
        "test_ew_sharpe": test_results.get("ew_sharpe", 0),
        "ann_return_pct": full_results["ann_return_pct"],
        "ann_vol_pct": full_results["ann_vol_pct"],
        "max_drawdown_pct": full_results["max_drawdown_pct"],
        "turnover": full_results["turnover"],
        "elapsed_sec": round(elapsed, 1),
        "loss_curve": {"train": [0, 0, 0, 0, 0], "val": [0, 0, 0, 0, 0]},
    }
    expected = {"val_sharpe": [0.5, 3.0], "train_time": [1, 30]}

    write_card(config, card_results, expected, verdict="BASELINE")

if __name__ == "__main__":
    main()
