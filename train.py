#!/usr/bin/env python3
"""
Exp 0: Linear baseline (DLinear-style)
Hypothesis: Simple linear map from price features to portfolio weights.
             This is the floor. Everything must beat this.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ── Config ──
SEED = 42
WINDOW = 60          # lookback days
REBAL_FREQ = 5       # rebalance every 5 days
TRAIN_RATIO = 0.7    # 70% train, 15% val, 15% test
VAL_RATIO = 0.15
LR = 1e-3
EPOCHS = 200
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TIME_BUDGET = 300    # 5 minutes

torch.manual_seed(SEED)
np.random.seed(SEED)

DATA = Path(__file__).parent / "data"


# ══════════════════════════════════════════════
# Feature Engineering (modify freely)
# ══════════════════════════════════════════════

def compute_features(d):
    """Compute price-based features from raw tensors. Returns (T, N_ASSETS, N_FEATURES)."""
    adj = d["adj_close"]   # (T, N)
    ret = d["log_return"]  # (T, N)
    T, N = ret.shape

    features = []

    # 1. Momentum (20-day log return)
    mom20 = ret.unfold(0, 20, 1).sum(dim=-1)  # (T-19, N)
    
    # 2. Realized vol (20-day)
    vol20 = ret.unfold(0, 20, 1).std(dim=-1)  # (T-19, N)

    # 3. Price / 200-day MA ratio (use adj_close)
    ma200 = adj.unfold(0, 200, 1).mean(dim=-1)  # (T-199, N)

    # 4. Price / 50-day MA ratio
    ma50 = adj.unfold(0, 50, 1).mean(dim=-1)  # (T-49, N)

    # Common length: need 200 days warmup
    L = T - 199
    start = 199

    feat_list = [
        ret[start:],                                           # 0: daily return
        mom20[start - 19:],                                    # 1: 20d momentum
        vol20[start - 19:],                                    # 2: 20d vol
        (adj[start:] / ma200 - 1.0),                          # 3: price/MA200 ratio
        (adj[start:] / ma50[start - 49:] - 1.0),              # 4: price/MA50 ratio
    ]

    # 5. 5-day momentum
    mom5 = ret.unfold(0, 5, 1).sum(dim=-1)
    feat_list.append(mom5[start - 4:])

    # 6. 60-day momentum
    mom60 = ret.unfold(0, 60, 1).sum(dim=-1)
    feat_list.append(mom60[start - 59:])

    # 7. Realized vol 60-day
    vol60 = ret.unfold(0, 60, 1).std(dim=-1)
    feat_list.append(vol60[start - 59:])

    # 8. High-low range (normalized)
    hl_range = (d["high"][start:] - d["low"][start:]) / adj[start:]
    feat_list.append(hl_range)

    # 9. Volume change (20d avg ratio)
    vol_data = d["volume"]
    vol_ma20 = vol_data.unfold(0, 20, 1).mean(dim=-1)
    vol_ratio = vol_data[start:] / (vol_ma20[start - 19:] + 1e-8)
    feat_list.append(vol_ratio)

    # Stack: (L, N, F)
    FEATURE_NAMES = ["daily_ret", "mom20", "vol20", "price_ma200", "price_ma50",
                     "mom5", "mom60", "vol60", "hl_range", "vol_ratio"]
    features = torch.stack(feat_list, dim=-1)  # (L, N, F)

    # Z-score normalize: expanding window (no look-ahead bias)
    # At each time t, normalize using mean/std from [0, t] only
    cum_sum = features.cumsum(dim=0)
    counts = torch.arange(1, T - 199 + 1, device=features.device).float().unsqueeze(-1).unsqueeze(-1)
    expanding_mean = cum_sum / counts
    cum_sq = (features ** 2).cumsum(dim=0)
    expanding_std = ((cum_sq / counts - expanding_mean ** 2).clamp(min=1e-8)).sqrt()
    # Need at least 20 days for stable stats
    features[20:] = (features[20:] - expanding_mean[20:]) / (expanding_std[20:] + 1e-8)
    features[:20] = 0  # not enough history, zero out

    return features, ret[start:], FEATURE_NAMES  # features, aligned returns, names


# ══════════════════════════════════════════════
# Model (modify freely)
# ══════════════════════════════════════════════

class LinearAllocator(nn.Module):
    """DLinear-style: flatten window features → linear → softmax weights."""
    def __init__(self, n_assets, n_features, window):
        super().__init__()
        self.n_assets = n_assets
        # Pool over window first, then linear
        self.pool_linear = nn.Linear(n_features, 1)  # per-asset feature aggregation
        self.out = nn.Linear(n_assets, n_assets)
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: (B, window, n_assets, n_features)
        B = x.shape[0]
        # Average over window, then aggregate features per asset
        x = x.mean(dim=1)  # (B, N, F)
        x = self.pool_linear(x).squeeze(-1)  # (B, N)
        logits = self.out(x) / self.temp.abs().clamp(min=0.1)
        weights = torch.softmax(logits, dim=-1)
        return weights


# ══════════════════════════════════════════════
# Loss & Metrics
# ══════════════════════════════════════════════

def portfolio_return(weights, future_returns):
    """weights: (B, N), future_returns: (B, N) → (B,)"""
    return (weights * future_returns).sum(dim=-1)

def sharpe_loss(port_returns):
    """Negative Sharpe ratio (to minimize)."""
    if port_returns.std() < 1e-8:
        return torch.tensor(0.0, device=port_returns.device)
    return -(port_returns.mean() / port_returns.std()) * math.sqrt(252)

def turnover_penalty(weights_seq, lam=0.001):
    """L1 turnover penalty."""
    if len(weights_seq) < 2:
        return torch.tensor(0.0)
    diffs = torch.stack([torch.abs(weights_seq[i] - weights_seq[i-1]).sum()
                         for i in range(1, len(weights_seq))])
    return lam * diffs.mean()


# ══════════════════════════════════════════════
# Data Pipeline
# ══════════════════════════════════════════════

def make_dataset(features, returns, window, rebal_freq):
    """Create (input, target) pairs. Rebalance every rebal_freq days."""
    T, N, F = features.shape
    X, Y = [], []
    for t in range(window, T - rebal_freq, rebal_freq):
        x = features[t - window:t]  # (window, N, F)
        # Target: average return over next rebal_freq days
        y = returns[t:t + rebal_freq].mean(dim=0)  # (N,)
        X.append(x)
        Y.append(y)
    return torch.stack(X), torch.stack(Y)


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def main():
    start_time = time.time()

    # Load
    d = torch.load(DATA / "tensors.pt", weights_only=False)
    features, returns, feat_names = compute_features(d)
    T, N, F = features.shape
    print(f"Features: {T} days × {N} assets × {F} features")

    # Dataset
    X, Y = make_dataset(features, returns, WINDOW, REBAL_FREQ)
    n_samples = len(X)
    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)

    X_train, Y_train = X[:n_train].to(DEVICE), Y[:n_train].to(DEVICE)
    X_val, Y_val = X[n_train:n_train + n_val].to(DEVICE), Y[n_train:n_train + n_val].to(DEVICE)
    X_test, Y_test = X[n_train + n_val:].to(DEVICE), Y[n_train + n_val:].to(DEVICE)

    print(f"Samples — train: {n_train}, val: {n_val}, test: {len(X_test)}")

    # Model
    model = LinearAllocator(N, F, WINDOW).to(DEVICE)
    MAX_PARAMS = 25_000
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if n_params > MAX_PARAMS:
        print(f"❌ ABORT: {n_params:,} params exceeds limit of {MAX_PARAMS:,}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Train
    best_val_sharpe = -999
    best_state = None

    for epoch in range(EPOCHS):
        if time.time() - start_time > TIME_BUDGET:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        w_train = model(X_train)
        port_ret = portfolio_return(w_train, Y_train)
        loss = sharpe_loss(port_ret)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                w_val = model(X_val)
                val_ret = portfolio_return(w_val, Y_val)
                val_sharpe = -(sharpe_loss(val_ret).item())

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

                if epoch % 50 == 0:
                    train_sharpe = -(loss.item())
                    print(f"Epoch {epoch:4d} | train_sharpe: {train_sharpe:.3f} | val_sharpe: {val_sharpe:.3f}")

    # Final eval on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        w_test = model(X_test)
        test_ret = portfolio_return(w_test, Y_test)
        test_sharpe = -(sharpe_loss(test_ret).item())

        # Max drawdown
        cum_ret = (1 + test_ret).cumprod(dim=0)
        peak = cum_ret.cummax(dim=0).values
        drawdown = (cum_ret - peak) / peak
        test_mdd = drawdown.min().item() * 100

        # Annual return
        n_periods = len(test_ret)
        total_ret = cum_ret[-1].item() - 1
        ann_ret = (1 + total_ret) ** (252 / (n_periods * REBAL_FREQ)) - 1

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"val_sharpe: {best_val_sharpe:.3f}")
    print(f"test_sharpe: {test_sharpe:.3f}")
    print(f"test_mdd: {test_mdd:.1f}%")
    print(f"test_ann_return: {ann_ret*100:.1f}%")
    print(f"params: {n_params:,}")
    print(f"time: {elapsed:.0f}s")
    print(f"{'='*50}")

    # Save experiment card
    config = {
        "model": model.__class__.__name__,
        "seed": SEED,
        "window": WINDOW,
        "rebal_freq": REBAL_FREQ,
        "lr": LR,
        "epochs": EPOCHS,
        "n_features": F,
        "features": feat_names,
        "n_assets": N,
        "n_params": n_params,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": len(X_test),
    }
    results = {
        "val_sharpe": round(best_val_sharpe, 4),
        "test_sharpe": round(test_sharpe, 4),
        "test_mdd": round(test_mdd, 2),
        "test_ann_return": round(ann_ret * 100, 2),
        "elapsed_sec": round(elapsed, 1),
    }
    # Load previous best for auto-verdict
    import json as _json
    prev_best = None
    cards_dir = Path(__file__).parent / "cards"
    for p in sorted(cards_dir.glob("exp_*.json")):
        with open(p) as f:
            c = _json.load(f)
            vs = c.get("results", {}).get("val_sharpe")
            if vs is not None and c.get("verdict") != "DISCARD":
                if prev_best is None or vs > prev_best:
                    prev_best = vs
    save_card(config, results, best_val_sharpe_so_far=prev_best)


def save_card(config, results, best_val_sharpe_so_far=None):
    """Save experiment card as JSON for reproducibility."""
    import json, datetime
    cards_dir = Path(__file__).parent / "cards"
    cards_dir.mkdir(exist_ok=True)

    # Find next experiment number
    existing = list(cards_dir.glob("exp_*.json"))
    n = len(existing)

    # Auto-verdict: compare with previous best
    verdict = "BASELINE"
    if best_val_sharpe_so_far is not None:
        verdict = "KEEP" if results["val_sharpe"] > best_val_sharpe_so_far else "DISCARD"

    card = {
        "exp": n,
        "timestamp": datetime.datetime.now().isoformat(),
        "verdict": verdict,
        "config": config,
        "results": results,
    }

    path = cards_dir / f"exp_{n:04d}.json"
    with open(path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Card saved: {path}")


if __name__ == "__main__":
    main()
