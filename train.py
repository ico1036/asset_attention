#!/usr/bin/env python3
"""
Exp 0031: Ensemble of 3 Models (Different Seeds)
Hypothesis: Single models are unstable. Ensemble averaging reduces variance and may reveal
robust regime patterns that individual models miss.
Expected: val_sharpe [0.3, 1.5], train_time [60, 300]
Regime check: Does ensemble produce more stable regime-aware allocation?
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


def train_one_split(model, X, Y, train_idx, val_idx, epochs=100, lr=1e-3, patience=20, entropy_lambda=0.1):
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
        row_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)
        mean_entropy = row_entropy.mean()
        loss = loss + entropy_lambda * mean_entropy
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


def analyze_ensemble_attention(models, X, dates, indices):
    """Analyze attention from ensemble - check if models disagree meaningfully."""
    for m in models:
        m.eval()
    with torch.no_grad():
        all_attn = []
        all_weights = []
        for model in models:
            w, attn = model(X, return_attn=True)
            all_weights.append(w)
            row_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)
            all_attn.append(row_entropy.mean(dim=2).mean().item())
        
        # Ensemble weights (average)
        ensemble_w = torch.stack(all_weights).mean(dim=0)
        
        # Weight disagreement across models (std across ensemble)
        weight_std = torch.stack(all_weights).std(dim=0).mean().item()
        
    years = [int(dates[idx][:4]) for idx in indices]
    crisis = {2008, 2009, 2020, 2022}
    calm = {2017, 2018, 2019}
    crisis_idx = [i for i, y in enumerate(years) if y in crisis]
    calm_idx = [i for i, y in enumerate(years) if y in calm]
    
    regime = {
        "avg_entropy": sum(all_attn) / len(all_attn),
        "weight_disagreement": weight_std,
    }
    
    if crisis_idx and calm_idx:
        crisis_w = ensemble_w[crisis_idx].mean(dim=0)
        calm_w = ensemble_w[calm_idx].mean(dim=0)
        shift = [(c - l) for c, l in zip(crisis_w.tolist(), calm_w.tolist())]
        regime["crisis_weights"] = crisis_w.tolist()
        regime["calm_weights"] = calm_w.tolist()
        regime["max_shift"] = max(abs(s) for s in shift)
    
    return regime


def main():
    t0 = time.time()
    d = load_data()
    ret = d["log_return"]
    dates = d["dates"]
    tickers = d["tickers"]
    T, N = ret.shape
    print(f"Data: {T} days × {N} assets")
    
    X, Y, indices = make_sliding_windows(ret)
    splits = make_expanding_splits(dates, indices)
    
    model = iTransformer(n_assets=N, d_model=8, n_patches=N_PATCHES)
    n_params = model.count_params()
    print(f"Model: iTransformer (ensemble of 3), {n_params} params each")
    assert n_params <= MAX_PARAMS
    
    all_test_weights, all_test_Y = [], []
    yearly_results = {}
    
    seeds = [42, 123, 456]
    
    for split in splits:
        models = []
        val_sharpes = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            model = iTransformer(n_assets=N, d_model=8, n_patches=N_PATCHES)
            val_s = train_one_split(model, X, Y, split["train"], split["val"], entropy_lambda=0.1)
            models.append(model)
            val_sharpes.append(val_s)
        
        # Ensemble prediction (average weights)
        with torch.no_grad():
            test_idx = split["test"]
            weights_list = [m(X[test_idx])[0] for m in models]
            w_test = torch.stack(weights_list).mean(dim=0)
            test_ret = (w_test * Y[test_idx]).sum(dim=-1)
            ts = sharpe(test_ret)
        
        year = split["test_year"]
        yearly_results[year] = {"val_sharpe": sum(val_sharpes) / len(val_sharpes), "test_sharpe": ts, "n_test": len(test_idx)}
        print(f"  {year}: val={val_sharpes[0]:.3f}/{val_sharpes[1]:.3f}/{val_sharpes[2]:.3f}, test={ts:.3f}")
        
        all_test_weights.append(w_test)
        all_test_Y.append(Y[test_idx])
    
    all_w = torch.cat(all_test_weights, dim=0)
    all_y = torch.cat(all_test_Y, dim=0)
    results = evaluate_and_print(all_w, all_y, "OOS_all", benchmark_n=N)
    regime = analyze_ensemble_attention(models, X, dates, indices)
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    
    config = {
        "model": "iTransformerEnsemble3",
        "n_params": n_params, "n_assets": N, "tickers": tickers,
        "d_model": 8, "n_patches": N_PATCHES, "temperature": 0.1,
        "entropy_lambda": 0.1, "ensemble_size": 3, "seeds": seeds,
        "architecture": "Ensemble of 3 iTransformers with different seeds",
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
    }
    write_card(config, card_results, {"val_sharpe": [0.3, 1.5], "train_time": [60, 300]})


if __name__ == "__main__":
    main()
