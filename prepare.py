#!/usr/bin/env python3
"""
Data prep + evaluation harness + constants. DO NOT MODIFY.

This file is the ground truth for:
  - Data loading and preparation
  - All evaluation metrics (Sharpe, Sortino, MDD, etc.)
  - Portfolio construction utilities (LW covariance, MinVar)
  - Experiment card writing
  - Constants (REBAL_FREQ, split ratios, etc.)

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

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "VEA", "VWO", "EEM",
           "TLT", "IEF", "SHY", "TIP", "HYG",
           "GLD", "DBC", "USO", "VNQ", "UUP"]
N_ASSETS = len(TICKERS)
START_DATE = "2007-08-01"

REBAL_FREQ = 5           # rebalance every N trading days
TRAIN_FRAC = 0.70         # train split
VAL_FRAC = 0.15           # val split (rest is test)
LOOKBACK_WINDOW = 60      # default lookback for features (trading days)
MAX_PARAMS = 25_000       # hard limit on learnable parameters
MIN_REBAL_PERIODS = 500   # minimum lookback start (skip first N days)

# Annualization: observations per year given REBAL_FREQ
OBS_PER_YEAR = 252 / REBAL_FREQ  # e.g., 50.4 for weekly
ANNUALIZE_FACTOR = math.sqrt(OBS_PER_YEAR)


# ═══════════════════════════════════════════════════════════════════════════
# Data preparation (one-time)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data():
    """parquet → tensors.pt. Run once: `uv run prepare.py`"""
    print("Loading parquet...")
    df = pd.read_parquet(DATA / "etf_daily.parquet")
    df = df.loc[START_DATE:]
    dates = df.index.get_level_values("date").unique().sort_values()
    dates = dates[dates >= START_DATE]

    adj_close = df["Adj Close"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill()
    log_ret = np.log(adj_close / adj_close.shift(1))
    log_ret = log_ret.iloc[1:]
    adj_close = adj_close.iloc[1:]

    open_px = df["Open"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill().iloc[1:]
    high_px = df["High"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill().iloc[1:]
    low_px  = df["Low"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill().iloc[1:]
    volume  = df["Volume"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill().iloc[1:]

    T = len(log_ret)
    print(f"Date range: {log_ret.index[0].date()} ~ {log_ret.index[-1].date()}")
    print(f"Shape: {T} days × {N_ASSETS} assets")

    tensors = {
        "adj_close": torch.tensor(adj_close.values, dtype=torch.float32),
        "log_return": torch.tensor(log_ret.values, dtype=torch.float32),
        "open": torch.tensor(open_px.values, dtype=torch.float32),
        "high": torch.tensor(high_px.values, dtype=torch.float32),
        "low": torch.tensor(low_px.values, dtype=torch.float32),
        "volume": torch.tensor(volume.values, dtype=torch.float32),
        "dates": log_ret.index.strftime("%Y-%m-%d").tolist(),
        "tickers": TICKERS,
    }
    torch.save(tensors, DATA / "tensors.pt")
    print(f"Saved tensors.pt: {T} days × {N_ASSETS} assets")
    print("Done.")


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (imported by train.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load tensors.pt. Returns dict with log_return, adj_close, dates, tickers, etc."""
    return torch.load(DATA / "tensors.pt", weights_only=False)


def make_samples(ret, rebal_freq=REBAL_FREQ, min_lookback=MIN_REBAL_PERIODS):
    """
    Create non-overlapping rebalancing samples from daily returns.

    Args:
        ret: (T, N) daily log returns
        rebal_freq: rebalance every N days
        min_lookback: skip first N days (for lookback window)

    Returns:
        sample_indices: list of ints (day index of each rebalance point)
        Y: (n_samples, N) period returns (SUM of log returns, not mean)
    """
    T, N = ret.shape
    sample_indices, Y_list = [], []
    for t in range(min_lookback, T - rebal_freq, rebal_freq):
        sample_indices.append(t)
        # SUM of log returns = log(cumulative return) over the period
        Y_list.append(ret[t:t + rebal_freq].sum(0))
    Y = torch.stack(Y_list)
    return sample_indices, Y


def split_data(n_samples, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """
    Time-series split (no shuffling). Returns (train_end, val_end).
    Train: [0, train_end), Val: [train_end, val_end), Test: [val_end, n_samples)
    """
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    return train_end, val_end


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation metrics (DO NOT CHANGE — these are the fixed metrics)
# ═══════════════════════════════════════════════════════════════════════════

def sharpe(returns):
    """
    Annualized Sharpe ratio from a series of period returns.

    Args:
        returns: 1D tensor of per-period returns (each period = REBAL_FREQ days)

    Returns:
        float: annualized Sharpe ratio (excess return assumed 0 for simplicity)

    Note: Annualization uses OBS_PER_YEAR = 252/REBAL_FREQ.
    For REBAL_FREQ=5: sqrt(50.4) ≈ 7.1, NOT sqrt(252).
    """
    if len(returns) < 2:
        return 0.0
    std = returns.std().item()
    if std < 1e-10:
        return 0.0
    return float(returns.mean().item() / std * ANNUALIZE_FACTOR)


def sortino(returns):
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    neg = returns[returns < 0]
    if len(neg) < 2:
        return float('inf') if returns.mean() > 0 else 0.0
    down_std = neg.std().item()
    if down_std < 1e-10:
        return 0.0
    return float(returns.mean().item() / down_std * ANNUALIZE_FACTOR)


def annualize_return(returns):
    """Annualized return from period returns (log space)."""
    return float(returns.mean().item() * OBS_PER_YEAR * 100)  # in percent


def annualize_vol(returns):
    """Annualized volatility from period returns."""
    return float(returns.std().item() * ANNUALIZE_FACTOR * 100)  # in percent


def max_drawdown(returns):
    """
    Maximum drawdown from period returns.
    Converts log returns to actual cumulative returns for accurate MDD.

    Returns:
        float: MDD as a negative percentage (e.g., -5.05 means -5.05%)
    """
    cum_log = returns.cumsum(0)
    cum_actual = torch.exp(cum_log) - 1.0  # actual cumulative return
    running_max = cum_actual.cummax(0)[0]
    drawdown = cum_actual - running_max
    return float(drawdown.min().item() * 100)


def turnover(weights):
    """
    Average per-period turnover.

    Args:
        weights: (T, N) portfolio weights over time

    Returns:
        float: mean absolute weight change per period
    """
    if len(weights) < 2:
        return 0.0
    diffs = (weights[1:] - weights[:-1]).abs().sum(dim=-1)
    return float(diffs.mean().item())


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio construction utilities
# ═══════════════════════════════════════════════════════════════════════════

def lw_cov(X):
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Args:
        X: (n, p) matrix of observations (e.g., daily returns)

    Returns:
        (p, p) shrunk covariance matrix
    """
    n, p = X.shape
    if n < 2:
        return torch.eye(p)

    S = torch.cov(X.T)
    mu = S.diagonal().mean()
    F = mu * torch.eye(p, device=X.device, dtype=X.dtype)

    X_c = X - X.mean(dim=0)
    # Vectorized sum of squared differences (replaces python for-loop)
    # sum_i (X_c[i]X_c[i]^T - S)^2
    outer_products = torch.einsum('bi,bj->bij', X_c, X_c)  # (n, p, p)
    diff_sq = ((outer_products - S.unsqueeze(0)) ** 2).sum(dim=0).sum()  # scalar
    delta = diff_sq / (n * n)
    gamma = ((F - S) ** 2).sum()

    if gamma < 1e-10:
        return S
    shrinkage = min(float(delta / gamma), 1.0)
    return (1 - shrinkage) * S + shrinkage * F


def minvar_weights(cov, n_assets=None):
    """
    Minimum variance portfolio weights (long-only).

    Args:
        cov: (N, N) covariance matrix
        n_assets: number of assets (inferred from cov if None)

    Returns:
        (N,) weight vector, long-only, sums to 1
    """
    N = cov.shape[0] if n_assets is None else n_assets
    try:
        # Use solve instead of inv for numerical stability
        ones = torch.ones(N, device=cov.device, dtype=cov.dtype)
        w = torch.linalg.solve(cov, ones)
        w = w.clamp(min=0.001)  # long-only constraint
        w = w / w.sum()
    except torch.linalg.LinAlgError:
        w = torch.ones(N, device=cov.device, dtype=cov.dtype) / N
    return w


def equal_weight(n_assets):
    """Equal weight benchmark."""
    return torch.ones(n_assets) / n_assets


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation harness (the equivalent of Karpathy's evaluate_bpb)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_strategy(weights, Y, split_name="full"):
    """
    Evaluate a portfolio strategy. This is the ground truth evaluation.

    Args:
        weights: (T, N) portfolio weights for each period
        Y: (T, N) period returns
        split_name: label for reporting

    Returns:
        dict with all metrics
    """
    assert weights.shape == Y.shape, f"Shape mismatch: weights {weights.shape} vs Y {Y.shape}"

    port_returns = (weights * Y).sum(dim=-1)  # (T,)

    return {
        "sharpe": sharpe(port_returns),
        "sortino": sortino(port_returns),
        "ann_return_pct": annualize_return(port_returns),
        "ann_vol_pct": annualize_vol(port_returns),
        "max_drawdown_pct": max_drawdown(port_returns),
        "turnover": turnover(weights),
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
    print(f"  Pos Prd:    {results['pos_periods_pct']:.1f}%")

    if benchmark_n is not None:
        ew = equal_weight(benchmark_n).unsqueeze(0).expand_as(weights)
        ew_results = evaluate_strategy(ew, Y, f"{split_name}_EW")
        print(f"  --- EW Benchmark ---")
        print(f"  EW Sharpe:  {ew_results['sharpe']:.3f}")
        print(f"  EW Return:  {ew_results['ann_return_pct']:.2f}%")
        print(f"  Delta:      {results['sharpe'] - ew_results['sharpe']:+.3f}")
        results["ew_sharpe"] = ew_results["sharpe"]
        results["delta_sharpe"] = results["sharpe"] - ew_results["sharpe"]

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment card writing
# ═══════════════════════════════════════════════════════════════════════════

def write_card(config, results, expected=None, verdict=None):
    """
    Write experiment card to cards/exp_NNNN.json.

    Args:
        config: dict with model name, n_params, etc.
        results: dict from evaluate_strategy + elapsed_sec
        expected: dict with expected ranges (optional)
        verdict: "KEEP" / "DISCARD" / "BASELINE" (optional, guard.py may override)

    Returns:
        card file path
    """
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
