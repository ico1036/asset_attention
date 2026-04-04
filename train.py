#!/usr/bin/env python3
"""
Exp 1: MLP with hidden layer + dropout + early stopping
Hypothesis: A single hidden layer MLP can capture non-linear feature interactions
             that the linear baseline misses. Dropout prevents overfitting.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ── Config ──
SEED = 42
WINDOW = 60
REBAL_FREQ = 5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
LR = 3e-3
EPOCHS = 500
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TIME_BUDGET = 300

torch.manual_seed(SEED)
np.random.seed(SEED)

DATA = Path(__file__).parent / "data"


def compute_features(d):
    adj = d["adj_close"]
    ret = d["log_return"]
    T, N = ret.shape

    mom20 = ret.unfold(0, 20, 1).sum(dim=-1)
    vol20 = ret.unfold(0, 20, 1).std(dim=-1)
    ma200 = adj.unfold(0, 200, 1).mean(dim=-1)
    ma50 = adj.unfold(0, 50, 1).mean(dim=-1)
    mom5 = ret.unfold(0, 5, 1).sum(dim=-1)
    mom60 = ret.unfold(0, 60, 1).sum(dim=-1)
    vol60 = ret.unfold(0, 60, 1).std(dim=-1)

    start = 199
    feat_list = [
        ret[start:],
        mom20[start - 19:],
        vol20[start - 19:],
        (adj[start:] / ma200 - 1.0),
        (adj[start:] / ma50[start - 49:] - 1.0),
        mom5[start - 4:],
        mom60[start - 59:],
        vol60[start - 59:],
        (d["high"][start:] - d["low"][start:]) / adj[start:],
        d["volume"][start:] / (d["volume"].unfold(0, 20, 1).mean(dim=-1)[start - 19:] + 1e-8),
    ]

    FEATURE_NAMES = ["daily_ret", "mom20", "vol20", "price_ma200", "price_ma50",
                     "mom5", "mom60", "vol60", "hl_range", "vol_ratio"]
    features = torch.stack(feat_list, dim=-1)

    cum_sum = features.cumsum(dim=0)
    counts = torch.arange(1, features.shape[0] + 1, device=features.device).float().unsqueeze(-1).unsqueeze(-1)
    expanding_mean = cum_sum / counts
    cum_sq = (features ** 2).cumsum(dim=0)
    expanding_std = ((cum_sq / counts - expanding_mean ** 2).clamp(min=1e-8)).sqrt()
    features[20:] = (features[20:] - expanding_mean[20:]) / (expanding_std[20:] + 1e-8)
    features[:20] = 0

    return features, ret[start:], FEATURE_NAMES


class MLPAllocator(nn.Module):
    """MLP: window-avg features → hidden → output → softmax."""
    def __init__(self, n_assets, n_features, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.n_assets = n_assets
        # Per-asset feature processing
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # Cross-asset layer
        self.cross = nn.Linear(n_assets, n_assets)
        self.dropout = nn.Dropout(dropout)
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: (B, window, n_assets, n_features)
        x = x.mean(dim=1)  # (B, N, F)
        x = self.fc1(x)  # (B, N, hidden)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)  # (B, N)
        x = self.cross(x)  # (B, N)
        logits = x / self.temp.abs().clamp(min=0.1)
        return torch.softmax(logits, dim=-1)


def portfolio_return(weights, future_returns):
    return (weights * future_returns).sum(dim=-1)

def sharpe_loss(port_returns):
    if port_returns.std() < 1e-8:
        return torch.tensor(0.0, device=port_returns.device)
    return -(port_returns.mean() / port_returns.std()) * math.sqrt(252)

def make_dataset(features, returns, window, rebal_freq):
    T, N, F = features.shape
    X, Y = [], []
    for t in range(window, T - rebal_freq, rebal_freq):
        x = features[t - window:t]
        y = returns[t:t + rebal_freq].mean(dim=0)
        X.append(x)
        Y.append(y)
    return torch.stack(X), torch.stack(Y)


def main():
    start_time = time.time()

    d = torch.load(DATA / "tensors.pt", weights_only=False)
    features, returns, feat_names = compute_features(d)
    T, N, F = features.shape
    print(f"Features: {T} days × {N} assets × {F} features")

    X, Y = make_dataset(features, returns, WINDOW, REBAL_FREQ)
    n_samples = len(X)
    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)

    X_train, Y_train = X[:n_train].to(DEVICE), Y[:n_train].to(DEVICE)
    X_val, Y_val = X[n_train:n_train + n_val].to(DEVICE), Y[n_train:n_train + n_val].to(DEVICE)
    X_test, Y_test = X[n_train + n_val:].to(DEVICE), Y[n_train + n_val:].to(DEVICE)

    print(f"Samples — train: {n_train}, val: {n_val}, test: {len(X_test)}")

    model = MLPAllocator(N, F, hidden_dim=32, dropout=0.3).to(DEVICE)
    MAX_PARAMS = 25_000
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if n_params > MAX_PARAMS:
        print(f"❌ ABORT: {n_params:,} params exceeds limit of {MAX_PARAMS:,}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_sharpe = -999
    best_state = None
    patience = 50
    no_improve = 0

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                w_val = model(X_val)
                val_ret = portfolio_return(w_val, Y_val)
                val_sharpe = -(sharpe_loss(val_ret).item())

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 5

                if epoch % 50 == 0:
                    train_sharpe = -(loss.item())
                    print(f"Epoch {epoch:4d} | train_sharpe: {train_sharpe:.3f} | val_sharpe: {val_sharpe:.3f}")

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        w_test = model(X_test)
        test_ret = portfolio_return(w_test, Y_test)
        test_sharpe = -(sharpe_loss(test_ret).item())

        cum_ret = (1 + test_ret).cumprod(dim=0)
        peak = cum_ret.cummax(dim=0).values
        drawdown = (cum_ret - peak) / peak
        test_mdd = drawdown.min().item() * 100

        n_periods = len(test_ret)
        total_ret = cum_ret[-1].item() - 1
        ann_ret = (1 + total_ret) ** (252 / (n_periods * REBAL_FREQ)) - 1

        eq_w = torch.ones(N, device=DEVICE) / N
        eq_ret = (eq_w * Y_test).sum(dim=-1)
        eq_sharpe = -(sharpe_loss(eq_ret).item())
        spy_ret = Y_test[:, 0]
        spy_sharpe = -(sharpe_loss(spy_ret).item())

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"benchmark_equal_weight_sharpe: {eq_sharpe:.3f}")
    print(f"benchmark_spy_sharpe: {spy_sharpe:.3f}")
    print(f"val_sharpe: {best_val_sharpe:.3f}")
    print(f"test_sharpe: {test_sharpe:.3f}")
    print(f"test_mdd: {test_mdd:.1f}%")
    print(f"test_ann_return: {ann_ret*100:.1f}%")
    print(f"params: {n_params:,}")
    print(f"time: {elapsed:.0f}s")
    print(f"{'='*50}")

    config = {
        "model": "MLPAllocator",
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
        "hidden_dim": 32,
        "dropout": 0.3,
    }
    results = {
        "val_sharpe": round(best_val_sharpe, 4),
        "test_sharpe": round(test_sharpe, 4),
        "test_mdd": round(test_mdd, 2),
        "test_ann_return": round(ann_ret * 100, 2),
        "elapsed_sec": round(elapsed, 1),
        "benchmark_equal_weight_sharpe": round(eq_sharpe, 4),
        "benchmark_spy_sharpe": round(spy_sharpe, 4),
    }
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
    import json, datetime
    cards_dir = Path(__file__).parent / "cards"
    cards_dir.mkdir(exist_ok=True)
    existing = list(cards_dir.glob("exp_*.json"))
    n = len(existing)
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
    print(f"Card saved: {path} — Verdict: {verdict}")


if __name__ == "__main__":
    main()
