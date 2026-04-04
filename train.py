#!/usr/bin/env python3
"""
Exp 0008: iTransformer-style — assets as tokens with temporal self-attention
Hypothesis: Exp 0007 failed because per-channel self-attn lost asset identity.
            iTransformer treats each variable as a token: (B, N_assets, Lookback).
            Self-attention over ASSETS (spatial) while keeping full time series per token.
            Each asset token contains its full 60-day return history.
            Spatial attention learns asset relationships; time info via linear embedding.
Expected: val_sharpe [0.5, 1.5], differentiated attention per asset pair
Change: iTransformer architecture — variables as tokens, spatial self-attention.
"""

import time, math, torch, torch.nn as nn
from prepare import (
    load_data, make_sliding_windows, make_expanding_splits,
    evaluate_and_print, write_card, sharpe,
    N_ASSETS, REBAL_FREQ, MAX_PARAMS, LOOKBACK, PATCH_SIZE, N_PATCHES,
    TX_COST_BPS,
)


class iTransformerAllocator(nn.Module):
    """
    iTransformer-style: variables (assets) as tokens.
    
    Data journey:
    1. (B, 12, 4) patch returns → transpose → (B, 4, 12) — each asset is a token
    2. Linear 12→d_model: embed each asset's time series → (B, 4, d)
    3. Self-attention over 4 asset tokens:
       "How should I weight THIS asset given what OTHER assets are doing?"
    4. (B, 4, d) → Linear → (B, 4, 1) → softmax → weights
    
    Time is preserved WITHIN each token (12 patches → d_model embedding).
    Attention is SPATIAL (across assets) but time-informed.
    
    Wait — this violates "attention over time dimension." Need temporal attention.
    
    Revised: Two-stage. First temporal self-attention per asset (with asset embedding),
    then spatial aggregation.
    """
    def __init__(self, n_assets=4, d_model=8, n_patches=12, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.temperature = temperature
        self.n_patches = n_patches

        # Asset embedding — gives each asset identity
        self.asset_embed = nn.Embedding(n_assets, d_model)
        
        # Patch embedding
        self.patch_embed = nn.Linear(1, d_model)
        
        # Positional encoding for time patches
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # (12, d)

        # Temporal self-attention (per asset, but asset-aware via embedding)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Score projection
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, x, return_attn=False):
        B, P, N = x.shape  # (B, 12, 4)
        
        # Transpose: (B, 4, 12)
        x_t = x.permute(0, 2, 1)  # (B, N, P)
        
        # Embed patches per asset: (B, N, P, 1) → (B, N, P, d)
        h = self.patch_embed(x_t.unsqueeze(-1))  # (B, N, P, d)
        
        # Add positional encoding
        h = h + self.pe.unsqueeze(0).unsqueeze(0)  # broadcast (1, 1, P, d)
        
        # Add asset embedding: (N, d) → (1, N, 1, d)
        asset_ids = torch.arange(N, device=x.device)
        asset_emb = self.asset_embed(asset_ids)  # (N, d)
        h = h + asset_emb.unsqueeze(0).unsqueeze(2)  # (B, N, P, d)
        
        # Reshape for attention: (B*N, P, d)
        h_flat = h.reshape(B * N, P, self.d_model)
        
        # Temporal self-attention
        Q = self.W_q(h_flat)
        K = self.W_k(h_flat)
        V = self.W_v(h_flat)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.temperature * math.sqrt(self.d_model))
        attn = torch.softmax(scores, dim=-1)  # (B*N, P, P)
        
        context = torch.bmm(attn, V)  # (B*N, P, d)
        
        # Pool over time (use last patch = most recent)
        pooled = context[:, -1, :]  # (B*N, d)
        
        # Reshape: (B, N, d)
        pooled = pooled.reshape(B, N, self.d_model)
        
        # Score per asset
        logits = self.score_proj(pooled).squeeze(-1)  # (B, N)
        weights = torch.softmax(logits, dim=-1)
        
        # Reshape attn for analysis: (B, N, P, P)
        attn_reshaped = attn.reshape(B, N, P, P)
        
        return weights, attn_reshaped

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
        w, _ = model(X_train)
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

    tickers = ["SPY", "TLT", "GLD", "SHY"]
    years = [int(dates[idx][:4]) for idx in indices]

    # Per-asset row entropy of temporal self-attention
    row_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)  # (B, N, P)
    per_asset_entropy = row_entropy.mean(dim=2)  # (B, N)

    print(f"\n=== Temporal Self-Attention Analysis (with asset embeddings) ===")
    for i, t in enumerate(tickers):
        e = per_asset_entropy[:, i]
        print(f"  {t} entropy: mean={e.mean():.3f}, std={e.std():.3f}")

    # Check if attention differs between assets now
    # Compare attention patterns of last query patch (most recent)
    last_q_attn = attn[:, :, -1, :]  # (B, N, P) — last query's attention over keys
    for i, t in enumerate(tickers):
        avg = last_q_attn[:, i].mean(dim=0)
        print(f"  {t} last-patch attn: {[f'{a:.3f}' for a in avg.tolist()]}")

    # Portfolio weights
    print(f"\n  Weight mean: {[f'{v:.3f}' for v in w.mean(dim=0).tolist()]}")
    print(f"  Weight std:  {[f'{v:.3f}' for v in w.std(dim=0).tolist()]}")

    crisis = {2008, 2009, 2020}
    calm = {2017, 2018, 2019}
    
    regime = {"avg_entropy": float(per_asset_entropy.mean())}
    
    crisis_idx = [i for i, y in enumerate(years) if y in crisis]
    calm_idx = [i for i, y in enumerate(years) if y in calm]
    if crisis_idx and calm_idx:
        crisis_w = w[crisis_idx].mean(dim=0)
        calm_w = w[calm_idx].mean(dim=0)
        shift = [(c - l) for c, l in zip(crisis_w.tolist(), calm_w.tolist())]
        print(f"\n  Crisis weights: {[f'{v:.3f}' for v in crisis_w.tolist()]}")
        print(f"  Calm weights:   {[f'{v:.3f}' for v in calm_w.tolist()]}")
        print(f"  Shift:          {[f'{s:+.3f}' for s in shift]}")
        regime["crisis_weights"] = crisis_w.tolist()
        regime["calm_weights"] = calm_w.tolist()
        regime["max_shift"] = max(abs(s) for s in shift)

        # Per-asset entropy in crisis vs calm
        crisis_e = per_asset_entropy[crisis_idx].mean().item()
        calm_e = per_asset_entropy[calm_idx].mean().item()
        print(f"  Crisis avg entropy: {crisis_e:.3f}, Calm: {calm_e:.3f}, Diff: {crisis_e-calm_e:+.3f}")
        regime["crisis_entropy"] = crisis_e
        regime["calm_entropy"] = calm_e
        regime["diff"] = crisis_e - calm_e

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

    model = iTransformerAllocator(n_assets=N, d_model=8, n_patches=N_PATCHES)
    n_params = model.count_params()
    print(f"Model: iTransformerAllocator, {n_params} params")
    assert n_params <= MAX_PARAMS

    all_test_weights, all_test_Y = [], []
    yearly_results = {}

    for split in splits:
        torch.manual_seed(42)
        model = iTransformerAllocator(n_assets=N, d_model=8, n_patches=N_PATCHES)
        val_s = train_one_split(model, X, Y, split["train"], split["val"])
        model.eval()
        with torch.no_grad():
            test_idx = split["test"]
            w_test, _ = model(X[test_idx])
            test_ret = (w_test * Y[test_idx]).sum(dim=-1)
            ts = sharpe(test_ret)
        year = split["test_year"]
        yearly_results[year] = {"val_sharpe": val_s, "test_sharpe": ts, "n_test": len(test_idx)}
        print(f"  {year}: val={val_s:.3f}, test={ts:.3f}")
        all_test_weights.append(w_test)
        all_test_Y.append(Y[test_idx])

    all_w = torch.cat(all_test_weights, dim=0)
    all_y = torch.cat(all_test_Y, dim=0)

    results = evaluate_and_print(all_w, all_y, "OOS_all", benchmark_n=N)
    regime = analyze_attention(model, X, dates, indices)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    config = {
        "model": "iTransformerAllocator",
        "n_params": n_params, "n_assets": N, "tickers": d["tickers"],
        "d_model": 8, "n_heads": 1, "n_patches": N_PATCHES, "temperature": 0.1,
        "architecture": "asset embeddings + temporal self-attention per asset (last patch pool) → score → softmax",
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
    write_card(config, card_results, {"val_sharpe": [0.5, 1.5], "train_time": [30, 300]})


if __name__ == "__main__":
    main()
