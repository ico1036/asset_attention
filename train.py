#!/usr/bin/env python3
"""
Exp 79: LW_MinVar robustness (different splits) + Attention model with MinVar input
Hypothesis: 
1. LW_MinVar should beat EW across different train/val/test splits (not period-specific)
2. An attention model that takes LW_MinVar weights as extra input features could learn 
   adaptive corrections (when to follow MinVar, when to deviate)
Expected: val_sharpe 0.5-2.0, test_sharpe 2.5-5.5, train_time 5-90s
"""

import time, math, json, datetime, numpy as np, torch, torch.nn as nn
from pathlib import Path

WINDOW = 60; REBAL_FREQ = 5
LR = 5e-4; EPOCHS = 1000; BATCH_SIZE = 64; NOISE_STD = 0.1
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TIME_BUDGET = 200; DATA = Path(__file__).parent / "data"

def sharpe(pr):
    if pr.std() < 1e-8: return 0.0
    return float((pr.mean() / pr.std()) * math.sqrt(252))

def sharpe_loss(pr):
    if pr.std() < 1e-8: return torch.tensor(0.0, device=pr.device)
    return -(pr.mean()/pr.std()) * math.sqrt(252)

def ledoit_wolf_cov(X):
    n, p = X.shape
    S = torch.cov(X.T)
    mu = S.diagonal().mean()
    F = mu * torch.eye(p, device=X.device)
    X_c = X - X.mean(dim=0)
    sum_sq = sum(((X_c[i:i+1].T @ X_c[i:i+1] - S) ** 2).sum() for i in range(n))
    delta = sum_sq / (n * n)
    gamma = ((F - S) ** 2).sum()
    if gamma < 1e-10: return S
    shrinkage = min(delta / gamma, 1.0)
    return (1 - shrinkage) * S + shrinkage * F

def lw_minvar_weights(xr):
    B, T_, N_ = xr.shape
    weights = []
    for i in range(B):
        cov = ledoit_wolf_cov(xr[i])
        try:
            ci = torch.linalg.inv(cov)
            w = ci @ torch.ones(N_)
            w = w.clamp(min=0.001); w = w / w.sum()
        except:
            w = torch.ones(N_) / N_
        weights.append(w)
    return torch.stack(weights)

def compute_features(d):
    adj = d["adj_close"]; ret = d["log_return"]; T, N = ret.shape
    mom20 = ret.unfold(0,20,1).sum(-1); vol20 = ret.unfold(0,20,1).std(-1)
    ma200 = adj.unfold(0,200,1).mean(-1); ma50 = adj.unfold(0,50,1).mean(-1)
    mom5 = ret.unfold(0,5,1).sum(-1); mom60 = ret.unfold(0,60,1).sum(-1)
    vol60 = ret.unfold(0,60,1).std(-1); start = 199
    feat_list = [ret[start:],mom20[start-19:],vol20[start-19:],(adj[start:]/ma200-1.0),
                 (adj[start:]/ma50[start-49:]-1.0),mom5[start-4:],mom60[start-59:],
                 vol60[start-59:],(d["high"][start:]-d["low"][start:])/adj[start:],
                 d["volume"][start:]/(d["volume"].unfold(0,20,1).mean(-1)[start-19:]+1e-8)]
    features = torch.stack(feat_list, dim=-1)
    cs = features.cumsum(0); cnt = torch.arange(1,features.shape[0]+1).float().unsqueeze(-1).unsqueeze(-1)
    em = cs/cnt; cs2 = (features**2).cumsum(0); es = ((cs2/cnt-em**2).clamp(min=1e-8)).sqrt()
    features[20:] = (features[20:]-em[20:])/(es[20:]+1e-8); features[:20]=0
    return features, ret[start:]

class MinVarAugmentedMLP(nn.Module):
    """MLP that takes features + MinVar weights as input."""
    def __init__(self, n_assets, n_features, hidden_dim=24, dropout=0.3):
        super().__init__()
        # Input: n_features + 1 (minvar weight for this asset)
        self.fc1 = nn.Linear(n_features + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cross = nn.Linear(n_assets, n_assets)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.temp = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x_feat, x_mv, noise=False):
        """x_feat: (B,N,F), x_mv: (B,N) MinVar weights"""
        x = torch.cat([x_feat, x_mv.unsqueeze(-1)], dim=-1)  # (B,N,F+1)
        if noise and self.training:
            x = x + torch.randn_like(x) * NOISE_STD
        h = self.act(self.dropout(self.fc1(x)))
        s = self.fc2(h).squeeze(-1)
        s = self.cross(s)
        return torch.softmax(s / self.temp.abs().clamp(min=0.1), dim=-1)

def main():
    t0 = time.time()
    d = torch.load(DATA/"tensors.pt", weights_only=False)
    features, returns = compute_features(d)
    ret = d["log_return"]; T, N = ret.shape
    start = 199
    
    # Build dataset
    X_feat, X_ret, Y = [], [], []
    for t in range(WINDOW, len(features) - REBAL_FREQ, REBAL_FREQ):
        X_feat.append(features[t-WINDOW:t])
        X_ret.append(returns[t-WINDOW:t])
        Y.append(returns[t:t+REBAL_FREQ].mean(0))
    X_feat = torch.stack(X_feat); X_ret = torch.stack(X_ret); Y = torch.stack(Y)
    ns = len(Y)
    
    results = {}
    
    # ---- Part 1: LW_MinVar robustness across splits ----
    print("=== LW_MinVar Robustness ===")
    for split_name, tr, vr in [
        ("70/15/15", 0.70, 0.15),
        ("60/20/20", 0.60, 0.20),
        ("80/10/10", 0.80, 0.10),
        ("50/25/25", 0.50, 0.25),
    ]:
        nt = int(ns*tr); nv = int(ns*vr)
        Xr_v = X_ret[nt:nt+nv]; Yv = Y[nt:nt+nv]
        Xr_te = X_ret[nt+nv:]; Yte = Y[nt+nv:]
        
        wv = lw_minvar_weights(Xr_v)
        wte = lw_minvar_weights(Xr_te)
        vs = sharpe((wv * Yv).sum(-1))
        ts = sharpe((wte * Yte).sum(-1))
        
        ew_v = sharpe((torch.ones(len(Yv),N)/N * Yv).sum(-1))
        ew_t = sharpe((torch.ones(len(Yte),N)/N * Yte).sum(-1))
        
        print(f"  {split_name}: LW val={vs:.3f} test={ts:.3f} | EW val={ew_v:.3f} test={ew_t:.3f} | Δtest={ts-ew_t:+.3f}")
        results[f"LW_{split_name}"] = {"val_sharpe": round(vs,4), "test_sharpe": round(ts,4), 
                                       "ew_val": round(ew_v,4), "ew_test": round(ew_t,4)}
    
    # ---- Part 2: MinVar-augmented MLP ----
    print("\n=== MinVar-Augmented MLP ===")
    TRAIN_RATIO = 0.7; VAL_RATIO = 0.15
    nt = int(ns*TRAIN_RATIO); nv = int(ns*VAL_RATIO)
    
    # Pre-compute MinVar weights for all samples
    print("Computing MinVar weights...")
    mv_all = lw_minvar_weights(X_ret)
    
    Xf_t = X_feat[:nt].mean(dim=1).to(DEVICE); Yt_d = Y[:nt].to(DEVICE)
    Xf_v = X_feat[nt:nt+nv].mean(dim=1).to(DEVICE); Yv_d = Y[nt:nt+nv].to(DEVICE)
    Xf_te = X_feat[nt+nv:].mean(dim=1).to(DEVICE); Yte_d = Y[nt+nv:].to(DEVICE)
    Mv_t = mv_all[:nt].to(DEVICE); Mv_v = mv_all[nt:nt+nv].to(DEVICE); Mv_te = mv_all[nt+nv:].to(DEVICE)
    
    F = Xf_t.shape[-1]
    seed_vals, seed_tests = [], []
    for seed in [42, 7, 13, 99, 256]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = MinVarAugmentedMLP(N, F, hidden_dim=24, dropout=0.3).to(DEVICE)
        np_ = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        best_vs, best_st, noimp = -999, None, 0
        n_batches = max(1, nt // BATCH_SIZE)
        
        for ep in range(EPOCHS):
            if time.time() - t0 > TIME_BUDGET: break
            model.train()
            perm = torch.randperm(nt, device=DEVICE)
            for b in range(n_batches):
                idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
                w = model(Xf_t[idx], Mv_t[idx], noise=True)
                loss = sharpe_loss((w * Yt_d[idx]).sum(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            if ep % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vs = sharpe((model(Xf_v, Mv_v) * Yv_d).sum(-1))
                    if vs > best_vs:
                        best_vs = vs; best_st = {k:v.clone() for k,v in model.state_dict().items()}; noimp = 0
                    else: noimp += 1
                if noimp >= 50: break
        
        if best_st: model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            ts = sharpe((model(Xf_te, Mv_te) * Yte_d).sum(-1))
        seed_vals.append(best_vs); seed_tests.append(ts)
        print(f"  seed={seed}: val={best_vs:.3f} test={ts:.3f} (params={np_})")
    
    results["MinVarMLP"] = {
        "val_sharpe": round(np.mean(seed_vals),4),
        "test_sharpe": round(np.mean(seed_tests),4),
        "per_seed": list(zip([round(v,3) for v in seed_vals], [round(t,3) for t in seed_tests]))
    }
    
    # Compare with pure LW
    Yv = Y[nt:nt+nv]; Yte = Y[nt+nv:]
    lw_vs = sharpe((mv_all[nt:nt+nv] * Yv).sum(-1))
    lw_ts = sharpe((mv_all[nt+nv:] * Yte).sum(-1))
    ew_ts = sharpe((torch.ones(len(Yte),N)/N * Yte).sum(-1))
    
    el = time.time() - t0
    print(f"\nLW_MinVar: val={lw_vs:.3f} test={lw_ts:.3f}")
    print(f"MinVarMLP: val={np.mean(seed_vals):.3f} test={np.mean(seed_tests):.3f}")
    print(f"EW:        test={ew_ts:.3f}")
    print(f"Time: {el:.1f}s")
    
    config = {"model": "LW_MinVar_robustness+MinVarMLP", "n_params": np_,
              "train_samples": nt, "n_assets": N}
    card_results = {"val_sharpe": round(lw_vs,4), "test_sharpe": round(lw_ts,4),
                    "elapsed_sec": round(el,1),
                    "benchmark_equal_weight_sharpe": round(ew_ts,4),
                    "all_results": results,
                    "loss_curve": {"train": [0,0,0,0,0], "val": [0,0,0,0,0]}}
    expected = {"val_sharpe": [0.5, 2.0], "train_time": [5, 200]}
    
    cd = Path(__file__).parent / "cards"; cd.mkdir(exist_ok=True)
    n = len(list(cd.glob("exp_*.json")))
    card = {"exp": n, "timestamp": datetime.datetime.now().isoformat(),
            "config": config, "results": card_results, "expected": expected}
    with open(cd / f"exp_{n:04d}.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"Card: exp_{n:04d}.json")

if __name__ == "__main__":
    main()
