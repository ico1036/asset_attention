#!/usr/bin/env python3
"""
Exp 107: FINAL — Comprehensive walk-forward evaluation of LW_MinVar_w500
The definitive robustness test with multiple window sizes, rolling forward.
Expected: val_sharpe [0.5, 2.0], test_sharpe [3.0, 7.0], train_time [1, 60]
"""

import time, math, json, datetime, numpy as np, torch
from pathlib import Path

REBAL_FREQ = 5
DATA = Path(__file__).parent / "data"

def sharpe(pr):
    if pr.std() < 1e-8: return 0.0
    return float((pr.mean() / pr.std()) * math.sqrt(252))

def lw_cov(X):
    n, p = X.shape
    S = torch.cov(X.T); mu = S.diagonal().mean(); F = mu * torch.eye(p)
    X_c = X - X.mean(dim=0)
    sum_sq = sum(((X_c[i:i+1].T @ X_c[i:i+1] - S)**2).sum() for i in range(n))
    delta = sum_sq / (n*n); gamma = ((F-S)**2).sum()
    if gamma < 1e-10: return S
    return (1-min(float(delta/gamma),1.0))*S + min(float(delta/gamma),1.0)*F

def minvar(cov, N):
    try:
        ci = torch.linalg.inv(cov)
        w = ci @ torch.ones(N); w = w.clamp(min=0.001); w = w / w.sum()
    except: w = torch.ones(N) / N
    return w

def main():
    t0 = time.time()
    d = torch.load(DATA/"tensors.pt", weights_only=False)
    ret = d["log_return"]; T, N = ret.shape

    sample_indices, Y_list = [], []
    for t in range(500, T - REBAL_FREQ, REBAL_FREQ):
        sample_indices.append(t)
        Y_list.append(ret[t:t+REBAL_FREQ].mean(0))
    Y = torch.stack(Y_list); ns = len(Y)

    # Pre-compute MinVar weights for all samples with w500
    mv_all = torch.stack([minvar(lw_cov(ret[max(0,i-500):i]), N) for i in sample_indices])
    ew = torch.ones(N) / N

    # Walk-forward with varying test window sizes
    print("=== Walk-Forward Analysis ===")
    print(f"Total samples: {ns} ({ns*5/252:.1f} years)")

    for test_weeks in [13, 26, 52]:
        print(f"\n--- Test window: {test_weeks} weeks ({test_weeks/52:.1f}yr) ---")
        mv_wins, ew_wins, total = 0, 0, 0
        mv_sharpes, ew_sharpes, deltas = [], [], []
        
        for start in range(0, ns - test_weeks, test_weeks):
            end = min(start + test_weeks, ns)
            if end - start < 5: break
            total += 1
            
            yw = Y[start:end]
            mvw = mv_all[start:end]
            
            s_mv = sharpe((mvw * yw).sum(-1))
            s_ew = sharpe((torch.ones(end-start, N)/N * yw).sum(-1))
            
            mv_sharpes.append(s_mv)
            ew_sharpes.append(s_ew)
            deltas.append(s_mv - s_ew)
            
            if s_mv > s_ew: mv_wins += 1
            else: ew_wins += 1
        
        print(f"  Windows: {total}")
        print(f"  MinVar wins: {mv_wins}/{total} ({mv_wins/total*100:.0f}%)")
        print(f"  MinVar Sharpe: {np.mean(mv_sharpes):.3f} ± {np.std(mv_sharpes):.3f}")
        print(f"  EW Sharpe:     {np.mean(ew_sharpes):.3f} ± {np.std(ew_sharpes):.3f}")
        print(f"  Delta:         {np.mean(deltas):+.3f} ± {np.std(deltas):.3f}")
        print(f"  Delta range:   [{min(deltas):.3f}, {max(deltas):.3f}]")
        print(f"  Min MV Sharpe: {min(mv_sharpes):.3f}")

    # Year-by-year analysis
    print(f"\n=== Year-by-Year Performance ===")
    print(f"{'Year':>6s} {'MV_Sharpe':>10s} {'EW_Sharpe':>10s} {'Delta':>8s} {'Winner':>8s}")
    print("-" * 48)
    
    # Map sample indices to dates
    dates = d.get("dates", None)
    
    yearly_results = {}
    for i in range(ns):
        idx = sample_indices[i]
        year = 2007 + idx // 252  # approximate
        if year not in yearly_results:
            yearly_results[year] = {"mv_rets": [], "ew_rets": []}
        mr = (mv_all[i] * Y[i]).sum().item()
        er = (ew * Y[i]).sum().item()
        yearly_results[year]["mv_rets"].append(mr)
        yearly_results[year]["ew_rets"].append(er)
    
    for year in sorted(yearly_results.keys()):
        if len(yearly_results[year]["mv_rets"]) < 10: continue
        mv_r = torch.tensor(yearly_results[year]["mv_rets"])
        ew_r = torch.tensor(yearly_results[year]["ew_rets"])
        s_mv = sharpe(mv_r); s_ew = sharpe(ew_r)
        delta = s_mv - s_ew
        winner = "MinVar" if delta > 0 else "EW"
        print(f"{year:>6d} {s_mv:10.3f} {s_ew:10.3f} {delta:+8.3f} {winner:>8s}")

    # Overall statistics
    print(f"\n=== Summary Statistics (full period) ===")
    mv_rets = (mv_all * Y).sum(-1)
    ew_rets = (torch.ones(ns,N)/N * Y).sum(-1)
    
    for name, rets in [("MinVar_w500", mv_rets), ("EqualWeight", ew_rets)]:
        s = sharpe(rets)
        ann_ret = rets.mean().item() * 252 * 100
        ann_vol = rets.std().item() * math.sqrt(252) * 100
        cum = rets.cumsum(0)
        mdd = (cum - cum.cummax(0)[0]).min().item() * 100
        pos_weeks = (rets > 0).float().mean().item() * 100
        sortino = rets.mean() / rets[rets<0].std() * math.sqrt(252) if (rets<0).any() else 0
        print(f"  {name:15s}: Sharpe={s:.3f} Ret={ann_ret:.2f}% Vol={ann_vol:.2f}% MDD={mdd:.2f}% PosWeek={pos_weeks:.1f}% Sortino={float(sortino):.3f}")

    el = time.time() - t0
    print(f"\nTime: {el:.1f}s")

    config = {"model": "LW_MinVar_w500_final", "n_params": 0, "train_samples": ns, "n_assets": N}
    card_results = {"val_sharpe": sharpe(mv_rets[:int(ns*0.85)]),
                    "test_sharpe": sharpe(mv_rets[int(ns*0.85):]),
                    "elapsed_sec": round(el,1),
                    "loss_curve": {"train": [0,0,0,0,0], "val": [0,0,0,0,0]}}
    expected = {"val_sharpe": [0.5, 2.0], "train_time": [1, 60]}

    cd = Path(__file__).parent / "cards"; cd.mkdir(exist_ok=True)
    n = len(list(cd.glob("exp_*.json")))
    card = {"exp": n, "timestamp": datetime.datetime.now().isoformat(),
            "config": config, "results": card_results, "expected": expected}
    with open(cd / f"exp_{n:04d}.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"Card: exp_{n:04d}.json")

if __name__ == "__main__":
    main()
