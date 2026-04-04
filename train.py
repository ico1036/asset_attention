#!/usr/bin/env python3
"""
Exp 91: LW MinVar w250 robustness — multiple train/val/test splits
Hypothesis: LW_w250 is genuinely better, not just lucky on one test split
Expected: val_sharpe [0.3, 1.5], test_sharpe [2.0, 6.0], train_time [1, 60]
"""

import time, math, json, datetime, torch
from pathlib import Path

REBAL_FREQ = 5
DATA = Path(__file__).parent / "data"

def sharpe(pr):
    if pr.std() < 1e-8: return 0.0
    return float((pr.mean() / pr.std()) * math.sqrt(252))

def ledoit_wolf_cov(X):
    n, p = X.shape
    S = torch.cov(X.T)
    mu = S.diagonal().mean()
    F = mu * torch.eye(p)
    X_c = X - X.mean(dim=0)
    sum_sq = sum(((X_c[i:i+1].T @ X_c[i:i+1] - S)**2).sum() for i in range(n))
    delta = sum_sq / (n*n); gamma = ((F-S)**2).sum()
    if gamma < 1e-10: return S
    shrink = min(float(delta/gamma), 1.0)
    return (1-shrink)*S + shrink*F

def minvar_weights(ret, idx, window):
    w_start = max(0, idx - window)
    xr = ret[w_start:idx]
    N = ret.shape[1]
    if len(xr) < 20: return torch.ones(N) / N
    cov = ledoit_wolf_cov(xr)
    try:
        ci = torch.linalg.inv(cov)
        w = ci @ torch.ones(N); w = w.clamp(min=0.001); w = w / w.sum()
    except: w = torch.ones(N) / N
    return w

def main():
    t0 = time.time()
    d = torch.load(DATA/"tensors.pt", weights_only=False)
    ret = d["log_return"]; T, N = ret.shape; start = 199

    sample_indices, Y_list = [], []
    for t in range(start + 250, T - REBAL_FREQ, REBAL_FREQ):
        sample_indices.append(t)
        Y_list.append(ret[t:t+REBAL_FREQ].mean(0))
    Y = torch.stack(Y_list); ns = len(Y)

    # Multiple splits: vary train ratio from 50% to 80%
    splits = [
        ("50/25/25", 0.50, 0.25),
        ("60/20/20", 0.60, 0.20),
        ("70/15/15", 0.70, 0.15),
        ("80/10/10", 0.80, 0.10),
    ]

    windows = [60, 120, 250]
    
    print(f"Total samples: {ns}, assets: {N}")
    print(f"\n{'Split':12s} {'Window':>8s} {'Val':>8s} {'Test':>8s} {'MDD':>8s} {'EW_Test':>8s} {'Delta':>8s}")
    print("-" * 70)

    all_results = {}
    wf_summary = {}
    
    for split_name, tr, vr in splits:
        nt = int(ns * tr); nv = int(ns * vr)
        val_idx = sample_indices[nt:nt+nv]; te_idx = sample_indices[nt+nv:]
        Yv = Y[nt:nt+nv]; Yte = Y[nt+nv:]
        if len(Yte) < 5: continue

        ew_test = sharpe((torch.ones(len(Yte),N)/N * Yte).sum(-1))

        for w in windows:
            wv = torch.stack([minvar_weights(ret, i, w) for i in val_idx])
            wte = torch.stack([minvar_weights(ret, i, w) for i in te_idx])
            vs = sharpe((wv * Yv).sum(-1)); ts = sharpe((wte * Yte).sum(-1))
            cum = (wte * Yte).sum(-1).cumsum(0)
            mdd = (cum - cum.cummax(0)[0]).min().item()
            delta = ts - ew_test
            print(f"{split_name:12s} {w:>8d} {vs:8.3f} {ts:8.3f} {mdd:8.4f} {ew_test:8.3f} {delta:+8.3f}")
            key = f"{split_name}_w{w}"
            all_results[key] = {"val_sharpe": round(vs,4), "test_sharpe": round(ts,4),
                                "mdd": round(mdd,4), "ew_test": round(ew_test,4), "delta": round(delta,4)}

        # Walk-forward for w250 vs w60 in this split
        wf_w250, wf_w60, wf_ew = 0, 0, 0
        step = max(1, nv // 2)
        for wf_start in range(0, ns - nt - step, step):
            ws_start = wf_start; ws_end = ws_start + nt
            wf_end = min(ws_end + step, ns)
            if wf_end <= ws_end or wf_end > ns: break
            wf_idx = sample_indices[ws_end:wf_end]; yw = Y[ws_end:wf_end]
            if len(yw) < 3: break
            
            w250 = torch.stack([minvar_weights(ret, i, 250) for i in wf_idx])
            w60 = torch.stack([minvar_weights(ret, i, 60) for i in wf_idx])
            s250 = sharpe((w250 * yw).sum(-1))
            s60 = sharpe((w60 * yw).sum(-1))
            sew = sharpe((torch.ones(len(yw),N)/N * yw).sum(-1))
            if s250 > s60: wf_w250 += 1
            else: wf_w60 += 1

        total = wf_w250 + wf_w60
        if total > 0:
            wf_summary[split_name] = {"w250_wins": wf_w250, "w60_wins": wf_w60, "total": total}
            print(f"  Walk-forward w250 vs w60: {wf_w250}/{total} ({wf_w250/total*100:.0f}%)")

    el = time.time() - t0
    print(f"\nTime: {el:.1f}s")

    # Find best consistent result
    # Use 70/15/15 w250 as main
    main_key = "70/15/15_w250"
    main = all_results.get(main_key, list(all_results.values())[0])

    config = {"model": "LW_MinVar_w250_robust", "n_params": 0, "train_samples": int(ns*0.7), "n_assets": N}
    card_results = {"val_sharpe": main["val_sharpe"], "test_sharpe": main["test_sharpe"],
                    "elapsed_sec": round(el,1),
                    "benchmark_equal_weight_sharpe": main.get("ew_test", 2.76),
                    "all_results": all_results, "wf_summary": wf_summary,
                    "loss_curve": {"train": [0,0,0,0,0], "val": [0,0,0,0,0]}}
    expected = {"val_sharpe": [0.3, 1.5], "train_time": [1, 60]}

    cd = Path(__file__).parent / "cards"; cd.mkdir(exist_ok=True)
    n = len(list(cd.glob("exp_*.json")))
    card = {"exp": n, "timestamp": datetime.datetime.now().isoformat(),
            "config": config, "results": card_results, "expected": expected}
    with open(cd / f"exp_{n:04d}.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"Card: exp_{n:04d}.json")

if __name__ == "__main__":
    main()
