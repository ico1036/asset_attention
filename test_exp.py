#!/usr/bin/env python3
"""Quick test runner for single experiment"""
import sys, time, math, torch, torch.nn as nn
from prepare import (
    load_data, make_sliding_windows, make_expanding_splits,
    evaluate_and_print, write_card, sharpe,
    N_ASSETS, MAX_PARAMS, N_PATCHES,
)

class ContrastiveRegimeAttention(nn.Module):
    def __init__(self, n_assets=4, d_model=16, n_patches=12, n_heads=2, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_patches = n_patches
        self.patch_embed = nn.Linear(n_assets, d_model)
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_assets))
    def forward(self, x, return_attn=False):
        B, P, N = x.shape
        h = self.patch_embed(x) + self.pe.unsqueeze(0)
        h_out, attn_weights = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        logits = self.fc(h_out.mean(dim=1))
        weights = torch.softmax(logits, dim=-1)
        return (weights, attn_weights) if return_attn else (weights, None)
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def train_one_split(model, X, Y, train_idx, val_idx, epochs=100, lr=1e-3, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    best_val, best_state, wait = -float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        w, attn = model(X_train)
        port_ret = (w * Y_train).sum(dim=-1)
        loss = -(port_ret.mean() / (port_ret.std() + 1e-8))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                w_val, _ = model(X_val)
                val_ret = (w_val * Y_val).sum(dim=-1)
                vs = sharpe(val_ret)
            if vs > best_val:
                best_val, best_state, wait = vs, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience // 5:
                    break
    if best_state:
        model.load_state_dict(best_state)
    return best_val

def analyze_regime_signal(model, X, dates, indices):
    model.eval()
    with torch.no_grad():
        w, attn = model(X, return_attn=True)
        avg_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item() if attn is not None else 0.0
    years = [int(dates[idx][:4]) for idx in indices]
    crisis = {2008, 2009, 2020, 2022}
    calm = {2017, 2018, 2019}
    crisis_idx = [i for i, y in enumerate(years) if y in crisis]
    calm_idx = [i for i, y in enumerate(years) if y in calm]
    regime = {"avg_entropy": avg_entropy}
    if crisis_idx and calm_idx:
        crisis_w = w[crisis_idx].mean(dim=0)
        calm_w = w[calm_idx].mean(dim=0)
        shift = [abs(c - l) for c, l in zip(crisis_w.tolist(), calm_w.tolist())]
        regime["crisis_weights"] = crisis_w.tolist()
        regime["calm_weights"] = calm_w.tolist()
        regime["max_shift"] = max(shift)
    return regime

print("Loading data...")
d = load_data()
ret, dates, tickers = d["log_return"], d["dates"], d["tickers"]
X, Y, indices = make_sliding_windows(ret)
splits = make_expanding_splits(dates, indices)
print(f"Data: {ret.shape}, X: {X.shape}, Splits: {len(splits)}")

print("\n=== Exp 0050: ContrastiveRegimeAttention ===")
torch.manual_seed(42)
model = ContrastiveRegimeAttention(n_assets=4, d_model=16, n_heads=2)
print(f"Params: {model.count_params()}")

all_test_weights, all_test_Y, yearly_results = [], [], {}
for i, split in enumerate(splits):
    print(f"  Split {i+1}/{len(splits)}: {split['test_year']}", end=" ")
    val_s = train_one_split(model, X, Y, split["train"], split["val"])
    with torch.no_grad():
        test_idx = split["test"]
        w_test, _ = model(X[test_idx])
        test_ret = (w_test * Y[test_idx]).sum(dim=-1)
        ts = sharpe(test_ret)
    yearly_results[split["test_year"]] = {"val_sharpe": val_s, "test_sharpe": ts, "n_test": len(test_idx)}
    all_test_weights.append(w_test)
    all_test_Y.append(Y[test_idx])
    print(f"test_sharpe={ts:.3f}")

all_w = torch.cat(all_test_weights, dim=0)
all_y = torch.cat(all_test_Y, dim=0)
results = evaluate_and_print(all_w, all_y, "Exp50", benchmark_n=4)
regime = analyze_regime_signal(model, X, dates, indices)

config = {
    "model": "ContrastiveRegimeAttention_exp50",
    "n_params": model.count_params(),
    "n_assets": 4,
    "tickers": tickers,
    "architecture": "ContrastiveRegimeAttention d=16",
    "has_attention": True,
    "preserves_time": True,
    "regime_signal": regime,
    "seed": 42,
    "d_model": 16,
    "n_heads": 2,
}
card_results = {
    "val_sharpe": sum(r["val_sharpe"] for r in yearly_results.values()) / len(yearly_results),
    "test_sharpe": results["sharpe"],
    "ann_return_pct": results["ann_return_pct"],
    "ann_vol_pct": results["ann_vol_pct"],
    "max_drawdown_pct": results["max_drawdown_pct"],
    "turnover": results["turnover"],
    "yearly": yearly_results,
    "elapsed_sec": 0,
}
write_card(config, card_results, {"val_sharpe": [0.3, 1.5], "train_time": [60, 300]})
print(f"\nExp 50: sharpe={results['sharpe']:.3f}, shift={regime.get('max_shift', 0):.4%}")
