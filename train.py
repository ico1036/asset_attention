#!/usr/bin/env python3
"""
Exp 69: Final multi-seed comparison: Linear vs MLP vs Attention+Cross (5 seeds each)
Hypothesis: Fair comparison across architectures with proper training.
Expected: All models val 0.5-1.5, test near EW (2.5-2.8), train_time 60-280s
"""

import time, math, json, datetime, numpy as np, torch, torch.nn as nn
from pathlib import Path

WINDOW = 60; REBAL_FREQ = 5; TRAIN_RATIO = 0.7; VAL_RATIO = 0.15
LR = 5e-4; EPOCHS = 1200; BATCH_SIZE = 64
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TIME_BUDGET = 280; DATA = Path(__file__).parent / "data"
NOISE_STD = 0.1; SEEDS = [42, 7, 13, 99, 256]

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

class LinearAllocator(nn.Module):
    def __init__(self, n_assets, n_features):
        super().__init__()
        self.fc = nn.Linear(n_features, 1)
        self.cross = nn.Linear(n_assets, n_assets)
        self.temp = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, noise=False):
        x = x.mean(dim=1)
        if noise and self.training:
            x = x + torch.randn_like(x) * NOISE_STD
        s = self.fc(x).squeeze(-1)
        s = self.cross(s)
        return torch.softmax(s / self.temp.abs().clamp(min=0.1), dim=-1)

class MLPAllocator(nn.Module):
    def __init__(self, n_assets, n_features, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cross = nn.Linear(n_assets, n_assets)
        self.dropout = nn.Dropout(dropout)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.act = nn.GELU()
    def forward(self, x, noise=False):
        x = x.mean(dim=1)
        if noise and self.training:
            x = x + torch.randn_like(x) * NOISE_STD
        x = self.act(self.dropout(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        x = self.cross(x)
        return torch.softmax(x / self.temp.abs().clamp(min=0.1), dim=-1)

class AttentionCrossAllocator(nn.Module):
    def __init__(self, n_assets, n_features, d_model=12, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)
        self.cross = nn.Linear(n_assets, n_assets)
        self.dropout = nn.Dropout(dropout)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.scale = d_model ** -0.5
    def forward(self, x, noise=False):
        x = x.mean(dim=1)
        if noise and self.training:
            x = x + torch.randn_like(x) * NOISE_STD
        h = self.proj(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        attn = torch.softmax((q @ k.transpose(-2,-1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        h = self.norm(h + attn @ v)
        s = self.out(h).squeeze(-1)
        s = self.cross(s)
        return torch.softmax(s / self.temp.abs().clamp(min=0.1), dim=-1)

def sharpe_loss(pr):
    if pr.std() < 1e-8: return torch.tensor(0.0, device=pr.device)
    return -(pr.mean()/pr.std()) * math.sqrt(252)

def make_dataset(features, returns, window, rebal_freq):
    T,N,F = features.shape; X,Y = [],[]
    for t in range(window, T-rebal_freq, rebal_freq):
        X.append(features[t-window:t]); Y.append(returns[t:t+rebal_freq].mean(0))
    return torch.stack(X), torch.stack(Y)

def train_model(model, Xt, Yt, Xv, Yv, seed, t0):
    torch.manual_seed(seed); np.random.seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_vs, best_st, noimp = -999, None, 0
    nt = len(Xt); n_batches = max(1, nt // BATCH_SIZE)
    for ep in range(EPOCHS):
        if time.time()-t0 > TIME_BUDGET: break
        model.train()
        perm = torch.randperm(nt, device=DEVICE)
        for b in range(n_batches):
            idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            w = model(Xt[idx], noise=True)
            loss = sharpe_loss((w*Yt[idx]).sum(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                vs = -(sharpe_loss((model(Xv)*Yv).sum(-1)).item())
                if vs > best_vs:
                    best_vs = vs; best_st = {k:v.clone() for k,v in model.state_dict().items()}; noimp = 0
                else: noimp += 1
            if noimp >= 60: break
    if best_st: model.load_state_dict(best_st)
    return best_vs

def main():
    t0 = time.time()
    d = torch.load(DATA/"tensors.pt", weights_only=False)
    features, returns, feat_names = compute_features(d)
    T,N,F = features.shape
    X,Y = make_dataset(features, returns, WINDOW, REBAL_FREQ)
    ns = len(X); nt = int(ns*TRAIN_RATIO); nv = int(ns*VAL_RATIO)
    Xt,Yt = X[:nt].to(DEVICE), Y[:nt].to(DEVICE)
    Xv,Yv = X[nt:nt+nv].to(DEVICE), Y[nt:nt+nv].to(DEVICE)
    Xte,Yte = X[nt+nv:].to(DEVICE), Y[nt+nv:].to(DEVICE)
    print(f"Samples — train:{nt}, val:{nv}, test:{len(Xte)}")
    eqs = -(sharpe_loss((torch.ones(N,device=DEVICE)/N*Yte).sum(-1)).item())

    all_results = {}
    for name, make_model in [
        ("Linear", lambda: LinearAllocator(N, F).to(DEVICE)),
        ("MLP", lambda: MLPAllocator(N, F).to(DEVICE)),
        ("Attn+Cross", lambda: AttentionCrossAllocator(N, F, d_model=12).to(DEVICE)),
    ]:
        seed_results = {}
        for seed in SEEDS:
            model = make_model()
            np_ = sum(p.numel() for p in model.parameters())
            vs = train_model(model, Xt, Yt, Xv, Yv, seed, t0)
            model.eval()
            with torch.no_grad():
                ts = -(sharpe_loss((model(Xte)*Yte).sum(-1)).item())
            seed_results[seed] = {"val": round(vs,3), "test": round(ts,3)}
            print(f"{name} seed={seed}: val={vs:.3f} test={ts:.3f} (params={np_})")
        vals = [r["val"] for r in seed_results.values()]
        tests = [r["test"] for r in seed_results.values()]
        all_results[name] = {"per_seed": seed_results, "params": np_,
                             "val_mean": round(np.mean(vals),3), "val_med": round(np.median(vals),3),
                             "test_mean": round(np.mean(tests),3), "test_med": round(np.median(tests),3)}

    el = time.time()-t0
    print(f"\n{'='*50}")
    for name, r in all_results.items():
        print(f"{name:12s} ({r['params']:4d}p) val: {r['val_mean']:.3f}±{r['val_med']:.3f} | test: {r['test_mean']:.3f}±{r['test_med']:.3f}")
    print(f"{'EW':12s}         val: —       | test: {eqs:.3f}")
    print(f"Time: {el:.0f}s")

    config={"model":"MultiSeed_3way_comparison","seeds":SEEDS,"window":WINDOW,
            "rebal_freq":REBAL_FREQ,"lr":LR,"epochs":EPOCHS,"batch_size":BATCH_SIZE,
            "n_features":F,"features":feat_names,"n_assets":N,"n_params":692,
            "train_samples":nt}
    results={"val_sharpe":round(all_results["MLP"]["val_mean"],4),
             "test_sharpe":round(all_results["MLP"]["test_mean"],4),
             "elapsed_sec":round(el,1),"benchmark_equal_weight_sharpe":round(eqs,4),
             "comparison":all_results,
             "loss_curve":{"train":[],"val":[],"epochs":[]}}
    expected={"val_sharpe":[0.5, 1.5],"train_time":[60, 280]}
    cd=Path(__file__).parent/"cards"; cd.mkdir(exist_ok=True)
    n=len(list(cd.glob("exp_*.json")))
    card={"exp":n,"timestamp":datetime.datetime.now().isoformat(),"config":config,"results":results,
          "expected":expected}
    with open(cd/f"exp_{n:04d}.json","w") as f: json.dump(card,f,indent=2)
    print(f"Card: exp_{n:04d}.json")

if __name__ == "__main__":
    main()
