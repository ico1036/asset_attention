#!/usr/bin/env python3
"""
Exp 0018: Warm-Start Training
Hypothesis: Carrying model weights across expanding windows helps early years.
Change: Initialize each year's model from prior year's trained weights.
Note: Using 100 epochs for faster completion.
"""

import time, math, torch, torch.nn as nn
from prepare import (
    load_data, make_sliding_windows, make_expanding_splits,
    evaluate_and_print, write_card, sharpe,
    N_ASSETS, MAX_PARAMS, N_PATCHES,
)


class iTransformer(nn.Module):
    def __init__(self, n_assets=4, d_model=8, n_patches=12, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.temperature = temperature
        self.n_patches = n_patches
        self.asset_embed = nn.Embedding(n_assets, d_model)
        self.patch_embed = nn.Linear(1, d_model)
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, x, return_attn=False):
        B, P, N = x.shape
        x_t = x.permute(0, 2, 1)
        h = self.patch_embed(x_t.unsqueeze(-1))
        h = h + self.pe.unsqueeze(0).unsqueeze(0)
        asset_ids = torch.arange(N, device=x.device)
        asset_emb = self.asset_embed(asset_ids)
        h = h + asset_emb.unsqueeze(0).unsqueeze(2)
        h_flat = h.reshape(B * N, P, self.d_model)
        Q = self.W_q(h_flat)
        K = self.W_k(h_flat)
        V = self.W_v(h_flat)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.temperature * math.sqrt(self.d_model))
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, V)
        pooled = context[:, -1, :]
        pooled = pooled.reshape(B, N, self.d_model)
        logits = self.score_proj(pooled).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        attn_reshaped = attn.reshape(B, N, P, P)
        return weights, attn_reshaped

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_one_split(model, X, Y, train_idx, val_idx, epochs=100, lr=1e-3, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    best_val = -float('inf')
    best_state = None
    wait = 0
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
                best_val = vs
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience // 5:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, best_state


def analyze_attention(model, X, dates, indices):
    model.eval()
    with torch.no_grad():
        w, attn = model(X, return_attn=True)
    years = [int(dates[idx][:4]) for idx in indices]
    row_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)
    per_asset_entropy = row_entropy.mean(dim=2)
    regime = {"avg_entropy": float(per_asset_entropy.mean())}
    crisis = {2008, 2009, 2020}
    calm = {2017, 2018, 2019}
    crisis_idx = [i for i, y in enumerate(years) if y in crisis]
    calm_idx = [i for i, y in enumerate(years) if y in calm]
    if crisis_idx and calm_idx:
        crisis_w = w[crisis_idx].mean(dim=0)
        calm_w = w[calm_idx].mean(dim=0)
        shift = [(c - l) for c, l in zip(crisis_w.tolist(), calm_w.tolist())]
        regime["crisis_weights"] = crisis_w.tolist()
        regime["calm_weights"] = calm_w.tolist()
        regime["max_shift"] = max(abs(s) for s in shift)
        crisis_e = per_asset_entropy[crisis_idx].mean().item()
        calm_e = per_asset_entropy[calm_idx].mean().item()
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
    print(f"Data: {T} days × {N} assets")
    X, Y, indices = make_sliding_windows(ret)
    splits = make_expanding_splits(dates, indices)
    model = iTransformer(n_assets=N, d_model=8, n_patches=N_PATCHES)
    n_params = model.count_params()
    print(f"Model: iTransformerWarmStart, {n_params} params")
    assert n_params <= MAX_PARAMS
    
    all_test_weights, all_test_Y = [], []
    yearly_results = {}
    prev_state = None
    
    for split in splits:
        torch.manual_seed(42)
        model = iTransformer(n_assets=N, d_model=8, n_patches=N_PATCHES)
        if prev_state is not None:
            model.load_state_dict(prev_state)
        val_s, best_state = train_one_split(model, X, Y, split["train"], split["val"])
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
        prev_state = best_state
    
    all_w = torch.cat(all_test_weights, dim=0)
    all_y = torch.cat(all_test_Y, dim=0)
    results = evaluate_and_print(all_w, all_y, "OOS_all", benchmark_n=N)
    regime = analyze_attention(model, X, dates, indices)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    
    config = {
        "model": "iTransformerWarmStart",
        "n_params": n_params, "n_assets": N, "tickers": d["tickers"],
        "d_model": 8, "n_patches": N_PATCHES, "temperature": 0.1,
        "architecture": "iTransformer with warm-start training (carry weights year-to-year)",
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
    write_card(config, card_results, {"val_sharpe": [0.3, 1.5], "train_time": [20, 200]})


if __name__ == "__main__":
    main()
