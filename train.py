#!/usr/bin/env python3
"""
Exp 114: Amplify Regime Signal → Portfolio Variation
Hypothesis: Exp 113 achieved regime-dependent attention (crisis entropy 0.686 vs calm 0.863)
            but portfolio weights barely vary (std ~1%). The attention signal gets smoothed
            by the output layer. Fix: use attention output DIRECTLY as portfolio logits via
            a cross-attention mechanism where learned asset queries attend to time patches.
            This creates a direct path: time attention → per-asset score → softmax weights.
Expected: val_sharpe [-1.0, 1.5], portfolio weight std > 0.03, regime signal maintained
Change vs Exp 113: Replace self-attention + mean pool + linear out with cross-attention
                   where N_ASSETS learned queries attend to time patches directly.
"""

import time, math, torch, torch.nn as nn
from prepare import (
    load_data, make_sliding_windows, make_expanding_splits,
    evaluate_and_print, write_card, sharpe,
    N_ASSETS, REBAL_FREQ, MAX_PARAMS, LOOKBACK, PATCH_SIZE, N_PATCHES,
    TX_COST_BPS,
)


class CrossAttentionAllocator(nn.Module):
    """
    Cross-attention: learned asset queries attend to time patches.
    
    Data journey:
    1. (B, 12, 4) raw patch returns
    2. LayerNorm → (B, 12, 4) normalized to O(1)
    3. Linear 4→8 + sinusoidal PE → Keys/Values (B, 12, 8)
    4. Learned asset queries (4, 8) → Queries
    5. Cross-attention: each asset query attends to 12 time patches
       → "which time periods matter for THIS asset's allocation?"
    6. Output: (B, 4) scores → softmax → portfolio weights
    
    This creates DIRECT asset-specific temporal attention. Each asset can
    focus on different time patches. During crises, assets should attend
    to recent volatile patches; during calm, attention spreads out.
    """
    def __init__(self, n_assets, d_model=8, n_patches=12):
        super().__init__()
        self.d_model = d_model
        self.n_assets = n_assets
        
        self.input_norm = nn.LayerNorm(n_assets)
        self.proj = nn.Linear(n_assets, d_model)

        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        # Learned asset queries — each asset has its own query vector
        self.asset_queries = nn.Parameter(torch.randn(n_assets, d_model) * 0.1)
        
        # K, V projections from time patches
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_k.weight, gain=2.0)

        # Score projection: d_model → 1 per asset
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        
        # Normalize + project
        x_norm = self.input_norm(x)
        h = self.proj(x_norm) + self.pe  # (B, 12, 8)
        
        K = self.W_k(h)  # (B, 12, 8)
        V = self.W_v(h)  # (B, 12, 8)
        
        # Cross attention: asset queries (4, 8) × keys (B, 12, 8)
        Q = self.asset_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 8)
        
        scores = torch.bmm(Q, K.transpose(1, 2))  # (B, 4, 12)
        attn = torch.softmax(scores, dim=-1)  # (B, 4, 12) — per-asset attention over time
        
        # Each asset gets its own time-weighted context
        context = torch.bmm(attn, V)  # (B, 4, 8)
        
        # Score per asset
        logits = self.score_proj(context).squeeze(-1)  # (B, 4)
        weights = torch.softmax(logits, dim=-1)  # (B, 4)
        
        if return_attn:
            return weights, attn  # attn shape: (B, 4, 12) — per-asset temporal attention
        return weights

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_one_split(model, X, Y, train_idx, val_idx, epochs=500, lr=1e-3, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    best_val = -float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        w = model(X_train)
        port_ret = (w * Y_train).sum(dim=-1)
        mean_r = port_ret.mean()
        std_r = port_ret.std()
        loss = -(mean_r / (std_r + 1e-8))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                w_val = model(X_val)
                val_ret = (w_val * Y_val).sum(dim=-1)
                vs = sharpe(val_ret)
            if vs > best_val:
                best_val = vs
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience // 5:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def analyze_attention(model, X, dates, indices):
    model.eval()
    with torch.no_grad():
        w, attn = model(X, return_attn=True)
        # attn shape: (B, 4, 12) — per asset, per time patch

    # Per-asset entropy
    entropy_per_asset = -(attn * (attn + 1e-10).log()).sum(dim=-1)  # (B, 4)
    entropy = entropy_per_asset.mean(dim=-1)  # (B,) — avg across assets

    years = [int(dates[idx][:4]) for idx in indices]
    year_entropy = {}
    for i, y in enumerate(years):
        year_entropy.setdefault(y, []).append(entropy[i].item())

    print(f"\n=== Attention Entropy (max={math.log(12):.3f}) ===")
    print(f"  Overall: mean={entropy.mean():.3f}, std={entropy.std():.3f}")
    for y in sorted(year_entropy.keys()):
        vals = year_entropy[y]
        m = sum(vals) / len(vals)
        s = (sum((v - m)**2 for v in vals) / len(vals))**0.5
        print(f"  {y}: {m:.3f} ± {s:.3f}")

    # Per-asset attention patterns
    tickers = ["SPY", "TLT", "GLD", "SHY"]
    for i, t in enumerate(tickers):
        avg = attn[:, i].mean(dim=0)  # (12,) avg attention for this asset
        print(f"  {t} avg attn: {[f'{a:.3f}' for a in avg.tolist()]}")

    # Portfolio weight variation
    print(f"\n  Portfolio weight mean: {[f'{v:.3f}' for v in w.mean(dim=0).tolist()]}")
    print(f"  Portfolio weight std:  {[f'{v:.3f}' for v in w.std(dim=0).tolist()]}")

    crisis = {2008, 2009, 2020}
    calm = {2017, 2018, 2019}
    crisis_e = [e for y, es in year_entropy.items() if y in crisis for e in es]
    calm_e = [e for y, es in year_entropy.items() if y in calm for e in es]

    regime = {"avg_entropy": float(entropy.mean()), "entropy_std": float(entropy.std())}
    if crisis_e and calm_e:
        ca, cl = sum(crisis_e)/len(crisis_e), sum(calm_e)/len(calm_e)
        diff = ca - cl
        print(f"\n  Crisis: {ca:.3f}, Calm: {cl:.3f}, Diff: {diff:+.3f} "
              f"({'REGIME SIGNAL!' if abs(diff) > 0.05 else 'weak'})")
        regime.update({"crisis_entropy": ca, "calm_entropy": cl, "diff": diff})

    # Check if portfolio weights change between crisis and calm
    crisis_idx = [i for i, y in enumerate(years) if y in crisis]
    calm_idx = [i for i, y in enumerate(years) if y in calm]
    if crisis_idx and calm_idx:
        crisis_w = w[crisis_idx].mean(dim=0)
        calm_w = w[calm_idx].mean(dim=0)
        print(f"\n  Crisis weights: {[f'{v:.3f}' for v in crisis_w.tolist()]}")
        print(f"  Calm weights:   {[f'{v:.3f}' for v in calm_w.tolist()]}")
        print(f"  Weight shift:   {[f'{(c-l):+.3f}' for c, l in zip(crisis_w.tolist(), calm_w.tolist())]}")
        regime["crisis_weights"] = crisis_w.tolist()
        regime["calm_weights"] = calm_w.tolist()

    return regime


def main():
    t0 = time.time()
    torch.manual_seed(42)

    d = load_data()
    ret = d["log_return"]
    dates = d["dates"]
    T, N = ret.shape
    print(f"Data: {T} days × {N} assets ({d['tickers']})")

    X, Y, indices = make_sliding_windows(ret)
    print(f"Samples: {len(X)}")

    splits = make_expanding_splits(dates, indices)
    print(f"Expanding window: {len(splits)} splits")

    model = CrossAttentionAllocator(n_assets=N, d_model=8, n_patches=N_PATCHES)
    n_params = model.count_params()
    print(f"Model: CrossAttentionAllocator, {n_params} params")
    assert n_params <= MAX_PARAMS

    all_test_weights, all_test_Y = [], []
    yearly_results = {}

    for split in splits:
        model = CrossAttentionAllocator(n_assets=N, d_model=8, n_patches=N_PATCHES)
        torch.manual_seed(42)
        val_sharpe = train_one_split(model, X, Y, split["train"], split["val"])

        model.eval()
        with torch.no_grad():
            test_idx = split["test"]
            w_test = model(X[test_idx])
            test_ret = (w_test * Y[test_idx]).sum(dim=-1)
            ts = sharpe(test_ret)

        year = split["test_year"]
        yearly_results[year] = {"val_sharpe": val_sharpe, "test_sharpe": ts, "n_test": len(test_idx)}
        print(f"  {year}: val={val_sharpe:.3f}, test={ts:.3f}")

        all_test_weights.append(w_test)
        all_test_Y.append(Y[test_idx])

    all_weights = torch.cat(all_test_weights, dim=0)
    all_Y = torch.cat(all_test_Y, dim=0)

    results = evaluate_and_print(all_weights, all_Y, "OOS_all", benchmark_n=N)
    regime = analyze_attention(model, X, dates, indices)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    config = {
        "model": "CrossAttentionAllocator",
        "n_params": n_params, "n_assets": N, "tickers": d["tickers"],
        "d_model": 8, "n_heads": 1, "n_patches": N_PATCHES,
        "architecture": "learned asset queries → cross-attn over time patches → scores → softmax",
        "has_attention": True, "preserves_time": True,
        "regime_signal": regime,
    }
    card_results = {
        "val_sharpe": sum(r["val_sharpe"] for r in yearly_results.values()) / len(yearly_results),
        "test_sharpe": results["sharpe"],
        "ann_return_pct": results["ann_return_pct"],
        "ann_vol_pct": results["ann_vol_pct"],
        "max_drawdown_pct": results["max_drawdown_pct"],
        "turnover": results["turnover"],
        "yearly": yearly_results,
        "elapsed_sec": round(elapsed, 1),
        "loss_curve": {"train": [], "val": []},
    }
    write_card(config, card_results, {"val_sharpe": [-1.0, 1.5], "train_time": [10, 120]})


if __name__ == "__main__":
    main()
