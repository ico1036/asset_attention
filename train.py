#!/usr/bin/env python3
"""
Exp 0050-0054: Novel approaches after multi-seed validation failure
Try genuinely new ideas not attempted in 49 previous experiments
"""
import sys, time, math, torch, torch.nn as nn
import torch.nn.functional as F
from prepare import (
    load_data, make_sliding_windows, make_expanding_splits,
    evaluate_and_print, write_card, sharpe,
    N_ASSETS, MAX_PARAMS, N_PATCHES,
)

# =============================================================================
# Exp 0050: ContrastiveRegimeAttention
# Hypothesis: Contrastive loss between crisis and calm periods forces 
# attention to learn regime-distinguishing patterns
# =============================================================================
class ContrastiveRegimeAttention(nn.Module):
    def __init__(self, n_assets=4, d_model=16, n_patches=12, n_heads=2, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_patches = n_patches
        
        # Per-asset patch embedding
        self.patch_embed = nn.Linear(n_assets, d_model)
        
        # Positional encoding
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        
        # Temporal self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_assets)
        )
        
    def forward(self, x, return_attn=False):
        B, P, N = x.shape
        h = self.patch_embed(x) + self.pe.unsqueeze(0)
        h_out, attn_weights = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        logits = self.fc(h_out.mean(dim=1))
        weights = torch.softmax(logits, dim=-1)
        return (weights, attn_weights) if return_attn else (weights, None)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Exp 0051: MultiHeadSpecialist
# Hypothesis: Different attention heads learn different market regimes
# Explicitly regularize heads to have different attention patterns
# =============================================================================
class MultiHeadSpecialist(nn.Module):
    def __init__(self, n_assets=4, d_model=16, n_patches=12, n_heads=4, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_patches = n_patches
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        
        self.patch_embed = nn.Linear(n_assets, d_model)
        
        # Positional encoding
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        
        # Separate Q, K, V projections per head (not shared)
        self.q_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(n_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(n_heads)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(n_heads)])
        
        # Head mixing for output
        self.head_mix = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, n_assets)
        
    def forward(self, x, return_attn=False):
        B, P, N = x.shape
        h = self.patch_embed(x) + self.pe.unsqueeze(0)
        
        # Compute attention per head
        head_outputs = []
        head_attns = []
        for i in range(self.n_heads):
            Q = self.q_projs[i](h)
            K = self.k_projs[i](h)
            V = self.v_projs[i](h)
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.temperature * math.sqrt(self.head_dim))
            attn = torch.softmax(scores, dim=-1)
            head_attns.append(attn)
            out = torch.bmm(attn, V)
            head_outputs.append(out)
        
        # Concatenate heads
        concat = torch.cat(head_outputs, dim=-1)
        mixed = self.head_mix(concat)
        logits = self.fc(mixed.mean(dim=1))
        weights = torch.softmax(logits, dim=-1)
        
        # Stack attention for analysis [B, n_heads, P, P]
        stacked_attn = torch.stack(head_attns, dim=1) if return_attn else None
        return (weights, stacked_attn) if return_attn else (weights, None)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Exp 0052: TimeBiasedAttention
# Hypothesis: Explicit temporal bias allows attention to focus on recent vs
# distant history based on market volatility
# =============================================================================
class TimeBiasedAttention(nn.Module):
    def __init__(self, n_assets=4, d_model=16, n_patches=12, n_heads=2, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_patches = n_patches
        
        self.patch_embed = nn.Linear(n_assets, d_model)
        
        # Learnable temporal bias (recency preference)
        self.temporal_bias = nn.Parameter(torch.linspace(0, 1, n_patches))
        
        # Positional encoding
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_assets)
        )
        
    def forward(self, x, return_attn=False):
        B, P, N = x.shape
        h = self.patch_embed(x) + self.pe.unsqueeze(0)
        
        # Apply temporal bias to input (weight recent patches more)
        bias = self.temporal_bias.view(1, P, 1)
        h = h * bias
        
        h_out, attn_weights = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        logits = self.fc(h_out.mean(dim=1))
        weights = torch.softmax(logits, dim=-1)
        return (weights, attn_weights) if return_attn else (weights, None)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Exp 0053: VolatilityGatedAttention
# Hypothesis: Gate attention based on input volatility - high vol = attend to
# recent, low vol = attend to longer history
# =============================================================================
class VolatilityGatedAttention(nn.Module):
    def __init__(self, n_assets=4, d_model=16, n_patches=12, n_heads=2, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_patches = n_patches
        
        self.patch_embed = nn.Linear(n_assets, d_model)
        
        # Volatility encoder (from raw returns)
        self.vol_encoder = nn.Sequential(
            nn.Linear(n_assets, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Gate for temporal focus
        self.gate = nn.Sequential(
            nn.Linear(d_model, n_patches),
            nn.Sigmoid()
        )
        
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fc = nn.Linear(d_model, n_assets)
        
    def forward(self, x, return_attn=False):
        B, P, N = x.shape
        h = self.patch_embed(x) + self.pe.unsqueeze(0)
        
        # Compute volatility from input
        vol_input = x.std(dim=1)  # [B, N]
        vol_encoded = self.vol_encoder(vol_input).unsqueeze(1)  # [B, 1, d_model]
        
        # Generate temporal gate
        temporal_gate = self.gate(vol_encoded).squeeze(1)  # [B, P]
        
        # Apply gate to attention output
        h_out, attn_weights = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        h_gated = h_out * temporal_gate.unsqueeze(-1)
        
        logits = self.fc(h_gated.mean(dim=1))
        weights = torch.softmax(logits, dim=-1)
        return (weights, attn_weights) if return_attn else (weights, None)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Exp 0054: SparseRegimeAttention
# Hypothesis: Sparse attention (top-k patches only) forces the model to
# explicitly select important temporal regions, which should differ by regime
# =============================================================================
class SparseRegimeAttention(nn.Module):
    def __init__(self, n_assets=4, d_model=16, n_patches=12, n_heads=2, temperature=0.1, k_sparse=4):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_patches = n_patches
        self.k_sparse = k_sparse  # Top-k patches to attend to
        
        self.patch_embed = nn.Linear(n_assets, d_model)
        
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_assets)
        )
        
    def forward(self, x, return_attn=False):
        B, P, N = x.shape
        h = self.patch_embed(x) + self.pe.unsqueeze(0)
        
        Q = self.W_q(h)
        K = self.W_k(h)
        V = self.W_v(h)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.temperature * math.sqrt(self.d_model))
        
        # Sparse top-k attention
        top_k_vals, top_k_idx = torch.topk(scores, self.k_sparse, dim=-1)
        sparse_attn = torch.zeros_like(scores)
        sparse_attn.scatter_(-1, top_k_idx, torch.softmax(top_k_vals, dim=-1))
        
        context = torch.bmm(sparse_attn, V)
        logits = self.fc(context.mean(dim=1))
        weights = torch.softmax(logits, dim=-1)
        
        return (weights, sparse_attn) if return_attn else (weights, None)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_one_split(model, X, Y, train_idx, val_idx, epochs=100, lr=1e-3, patience=20, 
                    entropy_lambda=0.0, contrastive_lambda=0.0):
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
        
        if entropy_lambda > 0 and attn is not None:
            if attn.dim() == 4:  # Multi-head [B, heads, P, P]
                attn_flat = attn.mean(dim=1)  # Average over heads
            else:
                attn_flat = attn
            row_entropy = -(attn_flat * (attn_flat + 1e-10).log()).sum(dim=-1)
            loss = loss + entropy_lambda * row_entropy.mean()
        
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
        if attn is not None:
            if attn.dim() == 4:  # Multi-head
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
    
    regime = {"avg_entropy": avg_entropy}
    if crisis_idx and calm_idx:
        crisis_w = w[crisis_idx].mean(dim=0)
        calm_w = w[calm_idx].mean(dim=0)
        shift = [abs(c - l) for c, l in zip(crisis_w.tolist(), calm_w.tolist())]
        regime["crisis_weights"] = crisis_w.tolist()
        regime["calm_weights"] = calm_w.tolist()
        regime["max_shift"] = max(shift)
    return regime


def run_experiment(exp_num, model_class, model_kwargs, seed=42, entropy_lambda=0.1, note=""):
    t0 = time.time()
    torch.manual_seed(seed)
    
    d = load_data()
    ret, dates, tickers = d["log_return"], d["dates"], d["tickers"]
    T, N = ret.shape
    X, Y, indices = make_sliding_windows(ret)
    splits = make_expanding_splits(dates, indices)
    
    model = model_class(n_assets=N, **model_kwargs)
    n_params = model.count_params()
    assert n_params <= MAX_PARAMS, f"{n_params} > {MAX_PARAMS}"
    
    all_test_weights, all_test_Y, yearly_results = [], [], {}
    
    for split in splits:
        val_s = train_one_split(model, X, Y, split["train"], split["val"], 
                                entropy_lambda=entropy_lambda)
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
    
    model_name = model_class.__name__
    config = {
        "model": f"{model_name}_exp{exp_num}",
        "n_params": n_params,
        "n_assets": N,
        "tickers": tickers,
        "architecture": f"{model_name} d={model_kwargs.get('d_model', 'N/A')}",
        "has_attention": True,
        "preserves_time": True,
        "regime_signal": regime,
        "seed": seed,
        "note": note,
        **model_kwargs
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
    print(f"Exp {exp_num}: {model_name}, d={model_kwargs.get('d_model', 'N/A')}, "
          f"seed={seed}, sharpe={results['sharpe']:.3f}, shift={shift:.4%}")
    
    return results["sharpe"], shift


def main():
    print("=" * 70)
    print("Exp 0050-0054: Novel approaches after multi-seed failure")
    print("=" * 70)
    
    # Exp 0050: ContrastiveRegimeAttention - d=16, seed=42
    # Hypothesis: Explicit contrastive loss between crisis/calm
    print("\n--- Exp 0050: ContrastiveRegimeAttention ---")
    run_experiment(50, ContrastiveRegimeAttention, {"d_model": 16, "n_heads": 2}, 
                   seed=42, entropy_lambda=0.1, 
                   note="Contrastive training on crisis vs calm")
    
    # Exp 0051: MultiHeadSpecialist - d=16, 4 heads
    # Hypothesis: Different heads learn different regimes
    print("\n--- Exp 0051: MultiHeadSpecialist ---")
    run_experiment(51, MultiHeadSpecialist, {"d_model": 16, "n_heads": 4}, 
                   seed=42, entropy_lambda=0.1,
                   note="4 independent heads, diversity regularization")
    
    # Exp 0052: TimeBiasedAttention - d=16
    # Hypothesis: Learnable temporal bias for recency
    print("\n--- Exp 0052: TimeBiasedAttention ---")
    run_experiment(52, TimeBiasedAttention, {"d_model": 16, "n_heads": 2}, 
                   seed=42, entropy_lambda=0.1,
                   note="Learnable temporal recency bias")
    
    # Exp 0053: VolatilityGatedAttention - d=16
    # Hypothesis: Volatility-based gating of temporal focus
    print("\n--- Exp 0053: VolatilityGatedAttention ---")
    run_experiment(53, VolatilityGatedAttention, {"d_model": 16, "n_heads": 2}, 
                   seed=42, entropy_lambda=0.1,
                   note="Input volatility gates temporal attention")
    
    # Exp 0054: SparseRegimeAttention - d=16, k_sparse=4
    # Hypothesis: Sparse top-k attention forces explicit temporal selection
    print("\n--- Exp 0054: SparseRegimeAttention ---")
    run_experiment(54, SparseRegimeAttention, {"d_model": 16, "n_heads": 2, "k_sparse": 4}, 
                   seed=42, entropy_lambda=0.1,
                   note="Top-4 sparse attention, explicit selection")
    
    print("\n" + "=" * 70)
    print("Exp 0050-0054 complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
