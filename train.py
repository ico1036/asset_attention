#!/usr/bin/env python3
"""
Exp 0058-0062: d_model Scaling Experiment (주인님 Override)
Hypothesis: d_model 8-16 sweet spot captures both regime signal + Sharpe
Test d_model = [8, 12, 16, 24, 32] with multi-seed validation
"""
import sys, time, math, torch, torch.nn as nn
import torch.nn.functional as F
from prepare import (
    load_data, make_sliding_windows, make_expanding_splits,
    evaluate_and_print, write_card, sharpe,
    N_ASSETS, MAX_PARAMS, N_PATCHES,
)

# =============================================================================
# iTransformerScaler - configurable d_model for scaling experiments
# Based on Exp 44 (d=8, sharpe=1.121) and Exp 31/32 (d=4/16, regime signal)
# =============================================================================
class iTransformerScaler(nn.Module):
    def __init__(self, n_assets=4, d_model=8, n_patches=12, n_heads=2, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_patches = n_patches
        
        # Asset embeddings (spatial tokenization like iTransformer)
        self.asset_embed = nn.Linear(n_patches, d_model)
        
        # Temporal self-attention across patches
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Feedforward per asset
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer norm for stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Output projection to portfolio weights
        self.score_proj = nn.Linear(d_model, 1)
        
    def forward(self, x, return_attn=False):
        # x: [B, n_patches, n_assets]
        B, P, N = x.shape
        
        # Transpose for asset-as-token: [B, n_assets, n_patches]
        x_t = x.transpose(1, 2)
        
        # Embed each asset's patch sequence: [B, n_assets, d_model]
        h = self.asset_embed(x_t)
        
        # Temporal self-attention (attend across patches for each asset)
        h_attn, attn_weights = self.temporal_attn(h, h, h, need_weights=True, average_attn_weights=False)
        h = self.norm1(h + h_attn)
        
        # Feedforward
        h_ffn = self.ffn(h)
        h = self.norm2(h + h_ffn)
        
        # Score per asset -> softmax weights
        scores = self.score_proj(h).squeeze(-1)  # [B, n_assets]
        weights = torch.softmax(scores, dim=-1)
        
        return (weights, attn_weights) if return_attn else (weights, None)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_one_split(model, X, Y, train_idx, val_idx, epochs=150, lr=1e-3, patience=25, entropy_lambda=0.1):
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
        
        # Entropy regularization on attention weights
        if entropy_lambda > 0 and attn is not None:
            if attn.dim() == 4:  # [B, n_heads, N, N]
                attn_flat = attn.mean(dim=1)  # [B, N, N]
            else:
                attn_flat = attn
            # Average over assets
            row_entropy = -(attn_flat * (attn_flat + 1e-10).log()).sum(dim=-1).mean()
            loss = loss + entropy_lambda * row_entropy
        
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
    """Analyze regime signal: crisis vs calm portfolio weight shift"""
    model.eval()
    with torch.no_grad():
        w, attn = model(X, return_attn=True)
        if attn is not None:
            if attn.dim() == 4:
                attn_flat = attn.mean(dim=1)
            else:
                attn_flat = attn
            avg_entropy = -(attn_flat * (attn_flat + 1e-10).log()).sum(dim=-1).mean().item()
        else:
            avg_entropy = 0.0
    
    years = [int(dates[idx][:4]) for idx in indices]
    crisis = {2008, 2009, 2020, 2022}
    calm = {2017, 2018, 2019}
    crisis_idx = [i for i, y in enumerate(years) if y in crisis]
    calm_idx = [i for i, y in enumerate(years) if y in calm]
    
    regime = {"avg_entropy": avg_entropy, "n_crisis": len(crisis_idx), "n_calm": len(calm_idx)}
    if crisis_idx and calm_idx:
        crisis_w = w[crisis_idx].mean(dim=0)
        calm_w = w[calm_idx].mean(dim=0)
        shift = [abs(c - l) for c, l in zip(crisis_w.tolist(), calm_w.tolist())]
        regime["crisis_weights"] = [round(x, 4) for x in crisis_w.tolist()]
        regime["calm_weights"] = [round(x, 4) for x in calm_w.tolist()]
        regime["max_shift"] = max(shift)
        regime["shifts"] = [round(s, 4) for s in shift]
    return regime


def run_experiment(exp_num, d_model, seed=42, entropy_lambda=0.1):
    """Run single experiment with given d_model and seed"""
    t0 = time.time()
    torch.manual_seed(seed)
    
    d = load_data()
    ret, dates, tickers = d["log_return"], d["dates"], d["tickers"]
    T, N = ret.shape
    X, Y, indices = make_sliding_windows(ret)
    splits = make_expanding_splits(dates, indices)
    
    model = iTransformerScaler(n_assets=N, d_model=d_model, n_heads=2, temperature=0.1)
    n_params = model.count_params()
    
    if n_params > MAX_PARAMS:
        print(f"WARNING: {n_params} params exceeds MAX_PARAMS {MAX_PARAMS}")
    
    all_test_weights, all_test_Y, yearly_results = [], [], {}
    
    for split in splits:
        val_s = train_one_split(model, X, Y, split["train"], split["val"], entropy_lambda=entropy_lambda)
        with torch.no_grad():
            test_idx = split["test"]
            w_test, _ = model(X[test_idx])
            test_ret = (w_test * Y[test_idx]).sum(dim=-1)
            ts = sharpe(test_ret)
        
        yearly_results[split["test_year"]] = {"val_sharpe": val_s, "test_sharpe": ts, "n_test": len(test_idx)}
        all_test_weights.append(w_test)
        all_test_Y.append(Y[test_idx])
    
    all_w = torch.cat(all_test_weights, dim=0)
    all_y = torch.cat(all_test_Y, dim=0)
    results = evaluate_and_print(all_w, all_y, f"Exp{exp_num}", benchmark_n=N)
    
    regime = analyze_regime_signal(model, X, dates, indices)
    
    config = {
        "model": f"iTransformerScaler_exp{exp_num}",
        "n_params": n_params,
        "n_assets": N,
        "tickers": tickers,
        "architecture": f"iTransformerScaler d={d_model}",
        "d_model": d_model,
        "has_attention": True,
        "preserves_time": True,
        "regime_signal": regime,
        "seed": seed,
    }
    
    card_results = {
        "val_sharpe": sum(r["val_sharpe"] for r in yearly_results.values()) / len(yearly_results),
        "test_sharpe": results["sharpe"],
        "ann_return_pct": results["ann_return_pct"],
        "ann_vol_pct": results["ann_vol_pct"],
        "max_drawdown_pct": results["max_drawdown_pct"],
        "turnover": results["turnover"],
        "yearly": yearly_results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    write_card(config, card_results, {"val_sharpe": [0.3, 1.5], "train_time": [60, 300]})
    
    shift = regime.get('max_shift', 0)
    print(f"  -> sharpe={results['sharpe']:.3f}, shift={shift:.2%}, params={n_params}")
    
    return results["sharpe"], shift, regime


def run_multi_seed(exp_num, d_model, seeds=[42, 123, 456]):
    """Run same config with multiple seeds for robustness check"""
    print(f"\n{'='*60}")
    print(f"Exp {exp_num}: iTransformerScaler d_model={d_model}")
    print(f"Multi-seed validation: {seeds}")
    print(f"{'='*60}")
    
    results = []
    for seed in seeds:
        print(f"  Seed {seed}: ", end="", flush=True)
        s, shift, regime = run_experiment(exp_num, d_model, seed=seed)
        results.append({"seed": seed, "sharpe": s, "shift": shift, "regime": regime})
        exp_num += 1
    
    # Summary
    print(f"\n  --- Multi-seed Summary for d={d_model} ---")
    shifts = [r["shift"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    print(f"  Sharpe:  mean={sum(sharpes)/len(sharpes):.3f}, range=[{min(sharpes):.3f}, {max(sharpes):.3f}]")
    print(f"  Shift:   mean={sum(shifts)/len(shifts):.2%}, range=[{min(shifts):.2%}, {max(shifts):.2%}]")
    
    # Check for robust regime signal (>10% in all seeds)
    robust_regime = all(s > 0.10 for s in shifts)
    print(f"  Robust regime (>10% all seeds): {'YES ✓' if robust_regime else 'NO ✗'}")
    
    return results, robust_regime


def main():
    print("=" * 70)
    print("Exp 0058-0062: d_model Scaling Experiment (주인님 Override)")
    print("Hypothesis: d_model 8-16 sweet spot for regime + Sharpe")
    print("=" * 70)
    
    # Prepare data once
    print("\nPreparing data...")
    d = load_data()
    ret, dates, tickers = d["log_return"], d["dates"], d["tickers"]
    X, Y, indices = make_sliding_windows(ret)
    print(f"  Samples: {len(X)}, Assets: {len(tickers)}, Patches: {X.shape[1]}")
    
    # Test configurations: d_model = [8, 12, 16, 24, 32]
    configs = [
        (58, 8),   # Exp 58-60: d=8 with 3 seeds
        (61, 12),  # Exp 61-63: d=12 with 3 seeds  
        (64, 16),  # Exp 64-66: d=16 with 3 seeds
        (67, 24),  # Exp 67-69: d=24 with 3 seeds
        (70, 32),  # Exp 70-72: d=32 with 3 seeds
    ]
    
    all_results = {}
    any_robust_regime = False
    
    for base_exp, d_model in configs:
        results, robust = run_multi_seed(base_exp, d_model)
        all_results[d_model] = results
        if robust:
            any_robust_regime = True
            print(f"\n  >>> ROBUST REGIME DETECTED at d_model={d_model} <<<\n")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: d_model Scaling Experiment")
    print("=" * 70)
    print(f"{'d_model':<10} {'Sharpe (mean)':<15} {'Shift (mean)':<15} {'Robust >10%':<12}")
    print("-" * 70)
    for d_model, results in all_results.items():
        sharpes = [r["sharpe"] for r in results]
        shifts = [r["shift"] for r in results]
        robust = "YES ✓" if all(s > 0.10 for s in shifts) else "NO ✗"
        print(f"{d_model:<10} {sum(sharpes)/len(sharpes):.3f}          {sum(shifts)/len(shifts):.2%}          {robust}")
    
    print("-" * 70)
    if any_robust_regime:
        print("RESULT: Robust regime signal detected in at least one configuration!")
    else:
        print("RESULT: No robust regime signal detected across any d_model configuration.")
        print("        Critic termination recommendation stands.")
    print("=" * 70)


if __name__ == "__main__":
    main()
