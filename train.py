#!/usr/bin/env python3
"""
Exp 109: Micro-Patch Attention v3 — Pilot (4 assets, daily, sliding window, expanding window)
Hypothesis: Tiny attention (d=8, 1 head) over 12 weekly patches can learn regime-like patterns
            with ~4600 sliding window samples and expanding window evaluation.
Expected: val_sharpe [0.0, 0.8], test_sharpe [-0.5, 1.0], train_time [30, 300]
Regime check: attention weights should differ between crisis (2008,2020) and calm (2017) periods
"""

import time, math, torch, torch.nn as nn
from prepare import (
    load_data, make_sliding_windows, make_expanding_splits,
    evaluate_and_print, write_card, sharpe,
    N_ASSETS, REBAL_FREQ, MAX_PARAMS, LOOKBACK, PATCH_SIZE, N_PATCHES,
    TX_COST_BPS,
)

# ═══════════════════════════════════════════════════════════════════════════
# Model: Micro-Patch Attention
# ═══════════════════════════════════════════════════════════════════════════

class MicroPatchAttention(nn.Module):
    """
    Simplest possible attention-over-time for portfolio allocation.

    Input: (batch, 12 patches, 4 assets) — weekly avg returns
    Attention: single-head self-attention over 12 time steps
    Output: (batch, 4) — portfolio weights via softmax

    Data journey:
    1. Each patch = 5-day avg return for 4 assets → (12, 4)
    2. Linear project 4 → d_model=8
    3. Self-attention over 12 patches: Q, K, V all from projected patches
       → "which weeks matter for today's allocation decision?"
    4. Attention-weighted mean → (8,)
    5. Linear 8 → 4 → softmax → weights
    """
    def __init__(self, n_assets, d_model=8):
        super().__init__()
        self.d_model = d_model

        # Project asset returns to d_model
        self.proj = nn.Linear(n_assets, d_model)

        # Single-head attention (Q, K, V projections)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output
        self.out = nn.Linear(d_model, n_assets)

        self.scale = math.sqrt(d_model)

    def forward(self, x, return_attn=False):
        """
        Args:
            x: (batch, n_patches, n_assets)
            return_attn: if True, also return attention weights
        Returns:
            weights: (batch, n_assets) — portfolio weights (softmax)
            attn: (batch, n_patches, n_patches) — if return_attn
        """
        # Project: (B, 12, 4) → (B, 12, 8)
        h = self.proj(x)

        # Attention: Q, K, V all from h
        Q = self.W_q(h)  # (B, 12, 8)
        K = self.W_k(h)  # (B, 12, 8)
        V = self.W_v(h)  # (B, 12, 8)

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 12, 12)
        attn = torch.softmax(scores, dim=-1)  # (B, 12, 12)

        # Attention-weighted values
        context = torch.bmm(attn, V)  # (B, 12, 8)

        # Mean pool over time → (B, 8)
        summary = context.mean(dim=1)

        # Output weights
        logits = self.out(summary)  # (B, 4)
        weights = torch.softmax(logits, dim=-1)  # (B, 4)

        if return_attn:
            return weights, attn
        return weights

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_one_split(model, X, Y, train_idx, val_idx, epochs=500, lr=1e-3, patience=50):
    """Train on one expanding window split. Returns best val_sharpe and model state."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    best_val = -float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        w = model(X_train)  # (n_train, 4)
        port_ret = (w * Y_train).sum(dim=-1)  # (n_train,)

        # Loss: negative Sharpe with tx cost penalty
        w_diff = (w[1:] - w[:-1]).abs().sum(dim=-1)
        tx = w_diff.mean() * TX_COST_BPS / 10000
        loss = -sharpe(port_ret) + tx * 100  # scale tx penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Val check
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
    """Check if attention weights change across market periods."""
    model.eval()
    with torch.no_grad():
        _, attn = model(X, return_attn=True)  # (n_samples, 12, 12)

    # Attention entropy per sample (higher = more uniform, lower = more focused)
    entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean(dim=-1)  # (n_samples,)

    # Group by year
    years = [int(dates[idx][:4]) for idx in indices]
    year_entropy = {}
    for i, y in enumerate(years):
        if y not in year_entropy:
            year_entropy[y] = []
        year_entropy[y].append(entropy[i].item())

    print("\n=== Attention Entropy by Year ===")
    print("(Lower = more focused, Higher = more uniform)")
    for y in sorted(year_entropy.keys()):
        vals = year_entropy[y]
        print(f"  {y}: {sum(vals)/len(vals):.3f} (n={len(vals)})")

    # Compare crisis vs calm
    crisis_years = {2008, 2009, 2020}
    calm_years = {2017, 2018, 2019}
    crisis_e = [e for y, es in year_entropy.items() if y in crisis_years for e in es]
    calm_e = [e for y, es in year_entropy.items() if y in calm_years for e in es]

    if crisis_e and calm_e:
        crisis_avg = sum(crisis_e) / len(crisis_e)
        calm_avg = sum(calm_e) / len(calm_e)
        diff = crisis_avg - calm_avg
        print(f"\n  Crisis avg: {crisis_avg:.3f}")
        print(f"  Calm avg:   {calm_avg:.3f}")
        print(f"  Difference: {diff:+.3f} ({'regime signal!' if abs(diff) > 0.05 else 'weak/no signal'})")
        return {"crisis_entropy": crisis_avg, "calm_entropy": calm_avg, "diff": diff}

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    torch.manual_seed(42)

    # Load pilot data
    d = load_data()
    ret = d["log_return"]
    dates = d["dates"]
    T, N = ret.shape
    print(f"Data: {T} days × {N} assets ({d['tickers']})")

    # Create sliding window samples
    X, Y, indices = make_sliding_windows(ret)
    print(f"Samples: {len(X)} (sliding window, lookback={LOOKBACK}, patch={PATCH_SIZE})")

    # Create expanding window splits
    splits = make_expanding_splits(dates, indices)
    print(f"Expanding window: {len(splits)} yearly splits")
    print(f"  First: train={splits[0]['train_size']}, test_year={splits[0]['test_year']}")
    print(f"  Last:  train={splits[-1]['train_size']}, test_year={splits[-1]['test_year']}")

    # Model
    model = MicroPatchAttention(n_assets=N, d_model=8)
    n_params = model.count_params()
    print(f"Model: MicroPatchAttention, {n_params} params")
    assert n_params <= MAX_PARAMS, f"Too many params: {n_params} > {MAX_PARAMS}"

    # Train with expanding window — collect test predictions per year
    all_test_weights = []
    all_test_Y = []
    all_test_indices = []
    yearly_results = {}

    for split in splits:
        # Re-initialize model for each year (PT-style)
        model = MicroPatchAttention(n_assets=N, d_model=8)
        torch.manual_seed(42)

        val_sharpe = train_one_split(model, X, Y, split["train"], split["val"])

        # Test
        model.eval()
        with torch.no_grad():
            test_idx = split["test"]
            w_test = model(X[test_idx])
            test_ret = (w_test * Y[test_idx]).sum(dim=-1)
            ts = sharpe(test_ret)

        year = split["test_year"]
        yearly_results[year] = {"val_sharpe": val_sharpe, "test_sharpe": ts, "n_test": len(test_idx)}
        print(f"  {year}: val={val_sharpe:.3f}, test={ts:.3f} (n={len(test_idx)})")

        all_test_weights.append(w_test)
        all_test_Y.append(Y[test_idx])
        all_test_indices.extend([indices[i] for i in test_idx])

    # Aggregate all OOS results
    all_weights = torch.cat(all_test_weights, dim=0)
    all_Y = torch.cat(all_test_Y, dim=0)

    print(f"\n=== Aggregate OOS Results ===")
    results = evaluate_and_print(all_weights, all_Y, "OOS_all", benchmark_n=N)

    # Attention analysis on last split
    regime = analyze_attention(model, X, dates, indices)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Write card
    config = {
        "model": "MicroPatchAttention",
        "n_params": n_params,
        "n_assets": N,
        "tickers": d["tickers"],
        "d_model": 8,
        "n_heads": 1,
        "n_patches": N_PATCHES,
        "patch_size": PATCH_SIZE,
        "lookback": LOOKBACK,
        "n_samples": len(X),
        "n_splits": len(splits),
        "has_attention": True,
        "preserves_time": True,
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
    expected = {"val_sharpe": [0.0, 0.8], "train_time": [30, 300]}

    write_card(config, card_results, expected)

if __name__ == "__main__":
    main()
