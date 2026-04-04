#!/usr/bin/env python3
"""
Exp 71: MLP learns DEVIATIONS from EW with "beat EW" loss
Hypothesis: Instead of learning absolute weights, learn small tilts on top of EW.
The loss directly penalizes underperforming EW: max(0, ew_ret - model_ret)^2.
This focuses the model on ONLY deviating from EW when confident.
Expected: val_sharpe 1.0-2.0, test_sharpe 2.5-3.5, train_time 30-90s
"""

import time, math, json, datetime, numpy as np, torch, torch.nn as nn
from pathlib import Path

WINDOW = 60; REBAL_FREQ = 5; TRAIN_RATIO = 0.7; VAL_RATIO = 0.15
LR = 5e-4; EPOCHS = 1500; BATCH_SIZE = 64
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TIME_BUDGET = 120; DATA = Path(__file__).parent / "data"
NOISE_STD = 0.1; SEED = 42
MAX_DEVIATION = 0.3  # max total deviation from EW

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
    NAMES = ["daily_ret","mom20","vol20","price_ma200","price_ma50","mom5","mom60","vol60","hl_range","vol_ratio"]
    features = torch.stack(feat_list, dim=-1)
    cs = features.cumsum(0); cnt = torch.arange(1,features.shape[0]+1).float().unsqueeze(-1).unsqueeze(-1)
    em = cs/cnt; cs2 = (features**2).cumsum(0); es = ((cs2/cnt-em**2).clamp(min=1e-8)).sqrt()
    features[20:] = (features[20:]-em[20:])/(es[20:]+1e-8); features[:20]=0
    return features, ret[start:], NAMES

class EWDeviationMLP(nn.Module):
    """Learns small deviations from Equal Weight."""
    def __init__(self, n_assets, n_features, hidden_dim=32, dropout=0.3, max_dev=0.3):
        super().__init__()
        self.n_assets = n_assets
        self.ew = 1.0 / n_assets
        self.max_dev = max_dev
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # per-asset deviation score
        self.cross = nn.Linear(n_assets, n_assets)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        # Scale parameter: how much to deviate from EW (learnable)
        self.scale = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, x, noise=False):
        x = x.mean(dim=1)  # (B, N, F)
        if noise and self.training:
            x = x + torch.randn_like(x) * NOISE_STD
        h = self.act(self.dropout(self.fc1(x)))  # (B, N, hidden)
        dev = self.fc2(h).squeeze(-1)  # (B, N) raw deviation scores
        dev = self.cross(dev)  # cross-asset mixing
        # Zero-mean deviations (so they sum to 0)
        dev = dev - dev.mean(dim=-1, keepdim=True)
        # Scale deviations (tanh to bound, then scale)
        dev = torch.tanh(dev) * self.scale.abs().clamp(max=self.max_dev / self.n_assets)
        # Final weights = EW + deviation, clamped positive
        w = self.ew + dev
        w = w.clamp(min=0.001)
        w = w / w.sum(dim=-1, keepdim=True)
        return w

def sharpe_loss(pr):
    if pr.std() < 1e-8: return torch.tensor(0.0, device=pr.device)
    return -(pr.mean()/pr.std()) * math.sqrt(252)

def beat_ew_loss(model_returns, ew_returns, alpha=0.5):
    """Combined: Sharpe of model + penalty for underperforming EW."""
    sharpe = sharpe_loss(model_returns)
    # Excess returns over EW
    excess = model_returns - ew_returns
    # Penalize periods where model underperforms EW
    underperformance = torch.clamp(-excess, min=0)
    penalty = underperformance.mean() * 100  # scale up
    return alpha * sharpe + (1 - alpha) * penalty

def make_dataset(features, returns, window, rebal_freq):
    T,N,F = features.shape; X,Y = [],[]
    for t in range(window, T-rebal_freq, rebal_freq):
        X.append(features[t-window:t]); Y.append(returns[t:t+rebal_freq].mean(0))
    return torch.stack(X), torch.stack(Y)

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    d = torch.load(DATA/"tensors.pt", weights_only=False)
    features, returns, feat_names = compute_features(d)
    T,N,F = features.shape
    X,Y = make_dataset(features, returns, WINDOW, REBAL_FREQ)
    ns = len(X); nt = int(ns*TRAIN_RATIO); nv = int(ns*VAL_RATIO)
    Xt,Yt = X[:nt].to(DEVICE), Y[:nt].to(DEVICE)
    Xv,Yv = X[nt:nt+nv].to(DEVICE), Y[nt:nt+nv].to(DEVICE)
    Xte,Yte = X[nt+nv:].to(DEVICE), Y[nt+nv:].to(DEVICE)
    print(f"Samples — train:{nt}, val:{nv}, test:{len(Xte)}")
    
    ew = torch.ones(N, device=DEVICE) / N
    ew_test = float((-(((ew*Yte).sum(-1).mean()/(ew*Yte).sum(-1).std())*math.sqrt(252))))
    ew_test = -ew_test
    
    # Multi-seed for robustness
    seeds = [42, 7, 13, 99, 256]
    all_val, all_test = [], []
    loss_curves = {"train": [], "val": []}
    
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = EWDeviationMLP(N, F, hidden_dim=32, dropout=0.3, max_dev=MAX_DEVIATION).to(DEVICE)
        np_ = sum(p.numel() for p in model.parameters())
        
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        best_vs, best_st, noimp = -999, None, 0
        n_batches = max(1, nt // BATCH_SIZE)
        train_losses, val_losses = [], []
        
        for ep in range(EPOCHS):
            if time.time() - t0 > TIME_BUDGET: break
            model.train()
            perm = torch.randperm(nt, device=DEVICE)
            ep_loss = 0
            for b in range(n_batches):
                idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
                w = model(Xt[idx], noise=True)
                pr = (w * Yt[idx]).sum(-1)
                ew_pr = (ew * Yt[idx]).sum(-1)
                loss = beat_ew_loss(pr, ew_pr, alpha=0.7)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
            sched.step()
            
            if ep % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vw = model(Xv)
                    vs = float(-sharpe_loss((vw * Yv).sum(-1)))
                    tl = ep_loss / n_batches
                    if seed == 42:
                        train_losses.append(tl)
                        val_losses.append(-vs)
                    if vs > best_vs:
                        best_vs = vs; best_st = {k:v.clone() for k,v in model.state_dict().items()}; noimp = 0
                    else: noimp += 1
                if noimp >= 60: break
        
        if best_st: model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            ts = float(-sharpe_loss((model(Xte) * Yte).sum(-1)))
            # Check deviation magnitude
            w_test = model(Xte)
            avg_dev = (w_test - ew).abs().mean().item()
        
        print(f"Seed {seed}: val={best_vs:.3f} test={ts:.3f} params={np_} avg_dev={avg_dev:.4f} scale={model.scale.item():.4f}")
        all_val.append(best_vs); all_test.append(ts)
    
    # Sample loss curve at 0%, 25%, 50%, 75%, 100%
    if train_losses:
        n = len(train_losses)
        idxs = [0, n//4, n//2, 3*n//4, n-1]
        loss_curves["train"] = [round(train_losses[i], 4) for i in idxs]
        loss_curves["val"] = [round(val_losses[i], 4) for i in idxs]
    
    el = time.time() - t0
    print(f"\nVal: mean={np.mean(all_val):.3f} med={np.median(all_val):.3f}")
    print(f"Test: mean={np.mean(all_test):.3f} med={np.median(all_test):.3f}")
    print(f"EW test: {ew_test:.3f}")
    print(f"Time: {el:.1f}s")
    
    config = {"model": "EWDeviationMLP", "n_params": np_, "seeds": seeds,
              "window": WINDOW, "rebal_freq": REBAL_FREQ, "lr": LR, "epochs": EPOCHS,
              "batch_size": BATCH_SIZE, "max_deviation": MAX_DEVIATION,
              "train_samples": nt, "n_assets": N, "n_features": F,
              "features": feat_names, "loss_type": "beat_ew_loss(alpha=0.7)"}
    results = {"val_sharpe": round(np.mean(all_val), 4),
               "test_sharpe": round(np.mean(all_test), 4),
               "val_median": round(np.median(all_val), 4),
               "test_median": round(np.median(all_test), 4),
               "per_seed_val": [round(v, 3) for v in all_val],
               "per_seed_test": [round(v, 3) for v in all_test],
               "elapsed_sec": round(el, 1),
               "benchmark_equal_weight_sharpe": round(ew_test, 4),
               "loss_curve": loss_curves}
    expected = {"val_sharpe": [1.0, 2.0], "train_time": [30, 120]}
    
    cd = Path(__file__).parent / "cards"; cd.mkdir(exist_ok=True)
    n = len(list(cd.glob("exp_*.json")))
    card = {"exp": n, "timestamp": datetime.datetime.now().isoformat(),
            "config": config, "results": results, "expected": expected}
    with open(cd / f"exp_{n:04d}.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"Card: exp_{n:04d}.json")

if __name__ == "__main__":
    main()
