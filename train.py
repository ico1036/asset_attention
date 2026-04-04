#!/usr/bin/env python3
"""
Exp 34: MLP with GELU activation
Hypothesis: GELU is smoother than ReLU, may find better minimum.
"""

import time, math, json, datetime, numpy as np, torch, torch.nn as nn
from pathlib import Path

SEED = 42; WINDOW = 60; REBAL_FREQ = 5; TRAIN_RATIO = 0.7; VAL_RATIO = 0.15
LR = 3e-3; EPOCHS = 500; DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TIME_BUDGET = 300; DATA = Path(__file__).parent / "data"

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

class MLPAllocator(nn.Module):
    def __init__(self, n_assets, n_features, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cross = nn.Linear(n_assets, n_assets)
        self.dropout = nn.Dropout(dropout)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.act = nn.GELU()
    def forward(self, x):
        x = x.mean(dim=1)
        x = self.act(self.dropout(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        x = self.cross(x)
        return torch.softmax(x / self.temp.abs().clamp(min=0.1), dim=-1)

def sharpe_loss(pr):
    if pr.std() < 1e-8: return torch.tensor(0.0, device=pr.device)
    return -(pr.mean()/pr.std()) * math.sqrt(252)

def make_dataset(features, returns, window, rebal_freq):
    T,N,F = features.shape; X,Y = [],[]
    for t in range(window, T-rebal_freq, rebal_freq):
        X.append(features[t-window:t]); Y.append(returns[t:t+rebal_freq].mean(0))
    return torch.stack(X), torch.stack(Y)

def main():
    t0 = time.time(); torch.manual_seed(SEED); np.random.seed(SEED)
    d = torch.load(DATA/"tensors.pt", weights_only=False)
    features, returns, feat_names = compute_features(d)
    T,N,F = features.shape
    X,Y = make_dataset(features, returns, WINDOW, REBAL_FREQ)
    ns = len(X); nt = int(ns*TRAIN_RATIO); nv = int(ns*VAL_RATIO)
    Xt,Yt = X[:nt].to(DEVICE), Y[:nt].to(DEVICE)
    Xv,Yv = X[nt:nt+nv].to(DEVICE), Y[nt:nt+nv].to(DEVICE)
    Xte,Yte = X[nt+nv:].to(DEVICE), Y[nt+nv:].to(DEVICE)
    print(f"Samples — train:{nt}, val:{nv}, test:{len(Xte)}")

    model = MLPAllocator(N, F).to(DEVICE)
    np_ = sum(p.numel() for p in model.parameters()); print(f"Params: {np_}")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_vs, best_st, noimp = -999, None, 0
    for ep in range(EPOCHS):
        if time.time()-t0 > TIME_BUDGET: break
        model.train(); w = model(Xt); loss = sharpe_loss((w*Yt).sum(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vs = -(sharpe_loss((model(Xv)*Yv).sum(-1)).item())
                if vs > best_vs: best_vs=vs; best_st={k:v.clone() for k,v in model.state_dict().items()}; noimp=0
                else: noimp+=5
            if noimp>=50: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        tr=(model(Xte)*Yte).sum(-1); ts=-(sharpe_loss(tr).item())
        cum=(1+tr).cumprod(0); pk=cum.cummax(0).values; mdd=((cum-pk)/pk).min().item()*100
        ar=(cum[-1].item())**(252/(len(tr)*REBAL_FREQ))-1
        eqs=-(sharpe_loss((torch.ones(N,device=DEVICE)/N*Yte).sum(-1)).item())
    el=time.time()-t0
    print(f"\n{'='*50}\neq:{eqs:.3f} val:{best_vs:.3f} test:{ts:.3f} mdd:{mdd:.1f}% ann:{ar*100:.1f}% t:{el:.0f}s")

    config={"model":"MLPAllocator_GELU","seed":SEED,"window":WINDOW,"rebal_freq":REBAL_FREQ,
            "lr":LR,"n_features":F,"features":feat_names,"n_assets":N,"n_params":np_,
            "hidden_dim":32,"dropout":0.3,"weight_decay":1e-3,"activation":"GELU"}
    results={"val_sharpe":round(best_vs,4),"test_sharpe":round(ts,4),"test_mdd":round(mdd,2),
             "test_ann_return":round(ar*100,2),"elapsed_sec":round(el,1),
             "benchmark_equal_weight_sharpe":round(eqs,4)}
    cd=Path(__file__).parent/"cards"; cd.mkdir(exist_ok=True)
    n=len(list(cd.glob("exp_*.json")))
    card={"exp":n,"timestamp":datetime.datetime.now().isoformat(),"config":config,"results":results}
    with open(cd/f"exp_{n:04d}.json","w") as f: json.dump(card,f,indent=2)
    print(f"Card: exp_{n:04d}.json")

if __name__ == "__main__":
    main()
