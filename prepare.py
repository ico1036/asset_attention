#!/usr/bin/env python3
"""
Data prep + evaluation harness + constants. DO NOT MODIFY.

This file is the ground truth for:
  - Data loading and preparation
  - All evaluation metrics (Sharpe, Sortino, MDD, etc.)
  - Portfolio construction utilities (LW covariance, MinVar)
  - Sliding window + expanding window data generation
  - Experiment card writing
  - Constants

train.py imports from here. The agent modifies ONLY train.py.
"""

import json, math, time, datetime
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Constants (fixed, do not modify)
# ═══════════════════════════════════════════════════════════════════════════

DATA = Path(__file__).parent / "data"
CARDS = Path(__file__).parent / "cards"

# Full universe (17 ETFs)
ALL_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "VEA", "VWO", "EEM",
               "TLT", "IEF", "SHY", "TIP", "HYG",
               "GLD", "DBC", "USO", "VNQ", "UUP"]

# Pilot universe (4 core asset classes)
PILOT_TICKERS = ["SPY", "TLT", "GLD", "SHY"]

# Active universe — change this to switch between pilot and full
TICKERS = PILOT_TICKERS
N_ASSETS = len(TICKERS)

START_DATE = "2007-08-01"

REBAL_FREQ = 1            # daily rebalancing for maximum data points
LOOKBACK = 60             # 60 trading days lookback window
PATCH_SIZE = 5            # 5 days per patch → 12 patches from 60 days
N_PATCHES = LOOKBACK // PATCH_SIZE  # 12
MAX_PARAMS = 25_000       # hard limit on learnable parameters
TX_COST_BPS = 2           # transaction cost in basis points

# Annualization
OBS_PER_YEAR = 252 / REBAL_FREQ  # 252 for daily
ANNUALIZE_FACTOR = math.sqrt(OBS_PER_YEAR)


# ═══════════════════════════════════════════════════════════════════════════
# Data preparation (one-time)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data():
    """parquet → tensors.pt. Run once: `uv run prepare.py`"""
    print("Loading parquet...")
    df = pd.read_parquet(DATA / "etf_daily.parquet")
    df = df.loc[START_DATE:]

    # Build for ALL tickers (superset), train.py selects subset
    tickers = ALL_TICKERS
    adj_close = df["Adj Close"].unstack("ticker")[tickers].loc[START_DATE:].ffill().bfill()
    log_ret = np.log(adj_close / adj_close.shift(1))
    log_ret = log_ret.iloc[1:]
    adj_close = adj_close.iloc[1:]

    open_px = df["Open"].unstack("ticker")[tickers].loc[START_DATE:].ffill().bfill().iloc[1:]
    high_px = df["High"].unstack("ticker")[tickers].loc[START_DATE:].ffill().bfill().iloc[1:]
    low_px  = df["Low"].unstack("ticker")[tickers].loc[START_DATE:].ffill().bfill().iloc[1:]
    volume  = df["Volume"].unstack("ticker")[tickers].loc[START_DATE:].ffill().bfill().iloc[1:]

    T = len(log_ret)
    print(f"Date range: {log_ret.index[0].date()} ~ {log_ret.index[-1].date()}")
    print(f"Shape: {T} days × {len(tickers)} assets")

    tensors = {
        "adj_close": torch.tensor(adj_close.values, dtype=torch.float32),
        "log_return": torch.tensor(log_ret.values, dtype=torch.float32),
        "open": torch.tensor(open_px.values, dtype=torch.float32),
        "high": torch.tensor(high_px.values, dtype=torch.float32),
        "low": torch.tensor(low_px.values, dtype=torch.float32),
        "volume": torch.tensor(volume.values, dtype=torch.float32),
        "dates": log_ret.index.strftime("%Y-%m-%d").tolist(),
        "tickers": tickers,
    }
    torch.save(tensors, DATA / "tensors.pt")
    print(f"Saved tensors.pt: {T} days × {len(tickers)} assets")
    print("Done.")


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data(tickers=None):
    """
    Load tensors.pt, optionally subset to specific tickers.

    Args:
        tickers: list of ticker strings to select. Default: TICKERS (pilot).

    Returns:
        dict with log_return (T, N_selected), dates, tickers, etc.
    """
    d = torch.load(DATA / "tensors.pt", weights_only=False)
    if tickers is None:
        tickers = TICKERS
    all_tickers = d["tickers"]
    indices = [all_tickers.index(t) for t in tickers]

    return {
        "log_return": d["log_return"][:, indices],
        "adj_close": d["adj_close"][:, indices],
        "dates": d["dates"],
        "tickers": tickers,
    }


def make_sliding_windows(ret, lookback=LOOKBACK, patch_size=PATCH_SIZE):
    """
    Create sliding window samples for attention training.
    Each sample: (N_PATCHES, N_ASSETS) — patch-averaged returns.

    Args:
        ret: (T, N) daily log returns
        lookback: total lookback in days (default 60)
        patch_size: days per patch (default 5)

    Returns:
        X: (n_samples, n_patches, n_assets) — input patches
        Y: (n_samples, n_assets) — next-day returns (target)
        indices: list of day indices for each sample
    """
    T, N = ret.shape
    n_patches = lookback // patch_size
    X_list, Y_list, indices = [], [], []

    for t in range(lookback, T - REBAL_FREQ):
        # Build patches: split lookback window into n_patches chunks
        window = ret[t - lookback:t]  # (lookback, N)
        patches = window.reshape(n_patches, patch_size, N).mean(dim=1)  # (n_patches, N)
        X_list.append(patches)
        # Target: next-period return
        Y_list.append(ret[t:t + REBAL_FREQ].sum(0))  # (N,)
        indices.append(t)

    X = torch.stack(X_list)  # (n_samples, n_patches, n_assets)
    Y = torch.stack(Y_list)  # (n_samples, n_assets)
    return X, Y, indices


def make_expanding_splits(dates, indices, val_pct=0.10):
    """
    Create expanding window train/val/test splits by year.
    PT-style: train on all data up to year Y, test on year Y+1.

    Args:
        dates: list of date strings from tensors
        indices: day indices from make_sliding_windows
        val_pct: fraction of training data for validation (from end)

    Returns:
        list of dicts: [{"train": [i...], "val": [i...], "test": [i...], "test_year": Y}, ...]
    """
    # Get year for each sample
    years = [int(dates[idx][:4]) for idx in indices]
    unique_years = sorted(set(years))

    # Need at least 3 years for first split
    if len(unique_years) < 4:
        raise ValueError(f"Need >= 4 years, got {len(unique_years)}")

    splits = []
    # First test year: 3rd year (need 2+ years for training)
    for test_year_idx in range(3, len(unique_years)):
        test_year = unique_years[test_year_idx]
        train_years = set(unique_years[:test_year_idx])

        train_idx = [i for i, y in enumerate(years) if y in train_years]
        test_idx = [i for i, y in enumerate(years) if y == test_year]

        if not train_idx or not test_idx:
            continue

        # Split last val_pct of training for validation
        val_size = max(1, int(len(train_idx) * val_pct))
        val_idx = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        splits.append({
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
            "test_year": test_year,
            "train_size": len(train_idx),
        })

    return splits


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation metrics (DO NOT CHANGE — these are the fixed metrics)
# ═══════════════════════════════════════════════════════════════════════════

def sharpe(returns, annual_factor=None):
    """
    Annualized Sharpe ratio.

    Args:
        returns: 1D tensor of per-period returns
        annual_factor: override annualization factor (default: ANNUALIZE_FACTOR)
    """
    if annual_factor is None:
        annual_factor = ANNUALIZE_FACTOR
    if len(returns) < 2:
        return 0.0
    std = returns.std().item()
    if std < 1e-10:
        return 0.0
    return float(returns.mean().item() / std * annual_factor)


def sortino(returns, annual_factor=None):
    """Annualized Sortino ratio (downside deviation only)."""
    if annual_factor is None:
        annual_factor = ANNUALIZE_FACTOR
    if len(returns) < 2:
        return 0.0
    neg = returns[returns < 0]
    if len(neg) < 2:
        return float('inf') if returns.mean() > 0 else 0.0
    down_std = neg.std().item()
    if down_std < 1e-10:
        return 0.0
    return float(returns.mean().item() / down_std * annual_factor)


def annualize_return(returns):
    """Annualized return from period returns (log space)."""
    return float(returns.mean().item() * OBS_PER_YEAR * 100)


def annualize_vol(returns):
    """Annualized volatility from period returns."""
    return float(returns.std().item() * ANNUALIZE_FACTOR * 100)


def max_drawdown(returns):
    """Maximum drawdown (converts log returns to actual cumulative)."""
    cum_log = returns.cumsum(0)
    cum_actual = torch.exp(cum_log) - 1.0
    running_max = cum_actual.cummax(0)[0]
    drawdown = cum_actual - running_max
    return float(drawdown.min().item() * 100)


def turnover(weights):
    """Average per-period turnover."""
    if len(weights) < 2:
        return 0.0
    diffs = (weights[1:] - weights[:-1]).abs().sum(dim=-1)
    return float(diffs.mean().item())


def tx_cost(weights, cost_bps=TX_COST_BPS):
    """Transaction cost from turnover."""
    return turnover(weights) * cost_bps / 10000


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio construction utilities (benchmarks)
# ═══════════════════════════════════════════════════════════════════════════

def lw_cov(X):
    """Ledoit-Wolf shrinkage covariance estimator."""
    n, p = X.shape
    if n < 2:
        return torch.eye(p)
    S = torch.cov(X.T)
    mu = S.diagonal().mean()
    F = mu * torch.eye(p, device=X.device, dtype=X.dtype)
    X_c = X - X.mean(dim=0)
    outer_products = torch.einsum('bi,bj->bij', X_c, X_c)
    diff_sq = ((outer_products - S.unsqueeze(0)) ** 2).sum(dim=0).sum()
    delta = diff_sq / (n * n)
    gamma = ((F - S) ** 2).sum()
    if gamma < 1e-10:
        return S
    shrinkage = min(float(delta / gamma), 1.0)
    return (1 - shrinkage) * S + shrinkage * F


def minvar_weights(cov, n_assets=None):
    """Minimum variance portfolio weights (long-only)."""
    N = cov.shape[0] if n_assets is None else n_assets
    try:
        ones = torch.ones(N, device=cov.device, dtype=cov.dtype)
        w = torch.linalg.solve(cov, ones)
        w = w.clamp(min=0.001)
        w = w / w.sum()
    except torch.linalg.LinAlgError:
        w = torch.ones(N, device=cov.device, dtype=cov.dtype) / N
    return w


def equal_weight(n_assets):
    """Equal weight benchmark."""
    return torch.ones(n_assets) / n_assets


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation harness
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_strategy(weights, Y, split_name="full"):
    """Evaluate a portfolio strategy. Ground truth evaluation."""
    assert weights.shape == Y.shape, f"Shape mismatch: weights {weights.shape} vs Y {Y.shape}"
    port_returns = (weights * Y).sum(dim=-1)
    return {
        "sharpe": sharpe(port_returns),
        "sortino": sortino(port_returns),
        "ann_return_pct": annualize_return(port_returns),
        "ann_vol_pct": annualize_vol(port_returns),
        "max_drawdown_pct": max_drawdown(port_returns),
        "turnover": turnover(weights),
        "tx_cost_pct": tx_cost(weights) * 100,
        "pos_periods_pct": float((port_returns > 0).float().mean().item() * 100),
        "n_periods": len(port_returns),
        "split": split_name,
    }


def evaluate_and_print(weights, Y, split_name="full", benchmark_n=None):
    """Evaluate and print results. Optionally compare to EW benchmark."""
    results = evaluate_strategy(weights, Y, split_name)
    print(f"\n=== {split_name} ({results['n_periods']} periods) ===")
    print(f"  Sharpe:     {results['sharpe']:.3f}")
    print(f"  Sortino:    {results['sortino']:.3f}")
    print(f"  Ann Return: {results['ann_return_pct']:.2f}%")
    print(f"  Ann Vol:    {results['ann_vol_pct']:.2f}%")
    print(f"  MDD:        {results['max_drawdown_pct']:.2f}%")
    print(f"  Turnover:   {results['turnover']:.4f}")
    print(f"  TX Cost:    {results['tx_cost_pct']:.4f}%")
    print(f"  Pos Prd:    {results['pos_periods_pct']:.1f}%")

    if benchmark_n is not None:
        ew = equal_weight(benchmark_n).unsqueeze(0).expand_as(weights)
        ew_results = evaluate_strategy(ew, Y, f"{split_name}_EW")
        print(f"  --- EW Benchmark ---")
        print(f"  EW Sharpe:  {ew_results['sharpe']:.3f}")
        print(f"  Delta:      {results['sharpe'] - ew_results['sharpe']:+.3f}")
        results["ew_sharpe"] = ew_results["sharpe"]
        results["delta_sharpe"] = results["sharpe"] - ew_results["sharpe"]
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment card writing
# ═══════════════════════════════════════════════════════════════════════════

def write_card(config, results, expected=None, verdict=None):
    """Write experiment card to cards/exp_NNNN.json."""
    CARDS.mkdir(exist_ok=True)
    n = len(list(CARDS.glob("exp_*.json")))
    card = {
        "exp": n,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "results": results,
    }
    if expected is not None:
        card["expected"] = expected
    if verdict is not None:
        card["verdict"] = verdict
    path = CARDS / f"exp_{n:04d}.json"
    with open(path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Card: {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Main (data prep only)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    prepare_data()
