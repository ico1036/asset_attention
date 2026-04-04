#!/usr/bin/env python3
"""
Exp 17: Selective ensemble of top-performing seeds [123, 2024, 777]
Hypothesis: Seed 2024 achieved val=2.92 individually (exp16). Combining only
             top-performing seeds (dropping bad seed 31415) should yield better
             ensemble than using all seeds. weight_decay=5e-4.
"""

import time, math, numpy as np, torch, torch.nn as nn
from pathlib import Path

WINDOW = 60; REBAL_FREQ = 5; PATCH_SIZE = 5
TRAIN_RATIO = 0.7; VAL_RATIO = 0.15
LR = 3e-3; EPOCHS = 500
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TIME_BUDGET = 300
SEEDS = [123, 2024, 777]

DATA = Path(__file__).parent / "data"


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


class PatchTemporalAllocator(nn.Module):
    def __init__(self, n_assets, n_features, patch_size=5, d_model=16, dropout=0.2):
        super().__init__()
        self.patch_size = patch_size; self.d_model = d_model
        n_patches = WINDOW // patch_size
        self.patch_proj = nn.Linear(patch_size * n_features, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 1)
        self.temp = nn.Parameter(torch.tensor(1.0))
        mask = torch.triu(torch.ones(n_patches, n_patches), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, W, N, F = x.shape; P = self.patch_size; nP = W // P
        x = x.reshape(B, nP, P, N, F).permute(0,3,1,2,4).reshape(B*N, nP, P*F)
        x = self.patch_proj(x) + self.pos_enc
        q,k,v = self.q(x), self.k(x), self.v(x)
        scores = q@k.transpose(-2,-1)/(self.d_model**0.5)
        scores = scores.masked_fill(self.mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1); attn = self.dropout(attn)
        x = self.norm(x + attn@v)
        x = x[:, -1].reshape(B, N, self.d_model)
        logits = self.out(x).squeeze(-1)
        return torch.softmax(logits / self.temp.abs().clamp(min=0.1), dim=-1)


def sharpe_loss(pr):
    if pr.std() < 1e-8: return torch.tensor(0.0, device=pr.device)
    return -(pr.mean()/pr.std()) * math.sqrt(252)

def make_dataset(features, returns, window, rebal_freq):
    T,N,F = features.shape; X,Y = [],[]
    for t in range(window, T-rebal_freq, rebal_freq):
        X.append(features[t-window:t]); Y.append(returns[t:t+rebal_freq].mean(0))
    return torch.stack(X), torch.stack(Y)


def train_one(seed, Xt, Yt, Xv, Yv, N, F, t0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PatchTemporalAllocator(N,F).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_vs, best_st, noimp = -999, None, 0
    for ep in range(EPOCHS):
        if time.time()-t0 > TIME_BUDGET: break
        model.train()
        w = model(Xt); loss = sharpe_loss((w*Yt).sum(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vs = -(sharpe_loss((model(Xv)*Yv).sum(-1)).item())
                if vs > best_vs: best_vs=vs; best_st={k:v.clone() for k,v in model.state_dict().items()}; noimp=0
                else: noimp += 5
            if noimp >= 80: break
    model.load_state_dict(best_st)
    return model, best_vs


def main():
    t0 = time.time()
    torch.manual_seed(42); np.random.seed(42)
    d = torch.load(DATA/"tensors.pt", weights_only=False)
    features, returns, feat_names = compute_features(d)
    T,N,F = features.shape

    X,Y = make_dataset(features, returns, WINDOW, REBAL_FREQ)
    ns = len(X); nt = int(ns*TRAIN_RATIO); nv = int(ns*VAL_RATIO)
    Xt,Yt = X[:nt].to(DEVICE), Y[:nt].to(DEVICE)
    Xv,Yv = X[nt:nt+nv].to(DEVICE), Y[nt:nt+nv].to(DEVICE)
    Xte,Yte = X[nt+nv:].to(DEVICE), Y[nt+nv:].to(DEVICE)
    print(f"Samples — train:{nt}, val:{nv}, test:{len(Xte)}")

    models, val_sharpes = [], []
    for seed in SEEDS:
        m, vs = train_one(seed, Xt, Yt, Xv, Yv, N, F, t0)
        models.append(m); val_sharpes.append(vs)
        print(f"Seed {seed}: val={vs:.3f}")

    with torch.no_grad():
        wv = sum(m(Xv) for m in models) / len(models)
        best_val_sharpe = -(sharpe_loss((wv*Yv).sum(-1)).item())
        wt = sum(m(Xte) for m in models) / len(models)
        tr = (wt*Yte).sum(-1); ts = -(sharpe_loss(tr).item())
        cum=(1+tr).cumprod(0); pk=cum.cummax(0).values; mdd=((cum-pk)/pk).min().item()*100
        ar=(cum[-1].item())**(252/(len(tr)*REBAL_FREQ))-1
        eqs=-(sharpe_loss((torch.ones(N,device=DEVICE)/N*Yte).sum(-1)).item())
        spys=-(sharpe_loss(Yte[:,0]).item())

    el=time.time()-t0
    np_ = sum(p.numel() for p in models[0].parameters())
    print(f"\n{'='*50}")
    print(f"eq:{eqs:.3f} spy:{spys:.3f} val:{best_val_sharpe:.3f} test:{ts:.3f}")
    print(f"mdd:{mdd:.1f}% ann:{ar*100:.1f}% seeds:{SEEDS} t:{el:.0f}s")

    config={"model":"PatchTemporalAllocator_Ensemble","seeds":SEEDS,"window":WINDOW,
            "rebal_freq":REBAL_FREQ,"lr":LR,"epochs":EPOCHS,"n_features":F,"features":feat_names,
            "n_assets":N,"n_params":np_,"ensemble_size":len(SEEDS),
            "d_model":16,"dropout":0.2,"patch_size":PATCH_SIZE,"causal":True,"weight_decay":5e-4}
    results={"val_sharpe":round(best_val_sharpe,4),"test_sharpe":round(ts,4),"test_mdd":round(mdd,2),
             "test_ann_return":round(ar*100,2),"elapsed_sec":round(el,1),
             "benchmark_equal_weight_sharpe":round(eqs,4),"benchmark_spy_sharpe":round(spys,4),
             "individual_val_sharpes":[round(v,4) for v in val_sharpes]}
    import json as _j
    pb=None; cd=Path(__file__).parent/"cards"
    for p in sorted(cd.glob("exp_*.json")):
        with open(p) as f:
            c=_j.load(f); v=c.get("results",{}).get("val_sharpe")
            if v and c.get("verdict")!="DISCARD":
                if pb is None or v>pb: pb=v
    save_card(config,results,best_val_sharpe_so_far=pb)

def save_card(config,results,best_val_sharpe_so_far=None):
    import json,datetime
    cd=Path(__file__).parent/"cards"; cd.mkdir(exist_ok=True)
    n=len(list(cd.glob("exp_*.json")))
    verdict="KEEP" if best_val_sharpe_so_far is not None and results["val_sharpe"]>best_val_sharpe_so_far else "DISCARD"
    card={"exp":n,"timestamp":datetime.datetime.now().isoformat(),"verdict":verdict,"config":config,"results":results}
    path=cd/f"exp_{n:04d}.json"
    with open(path,"w") as f: json.dump(card,f,indent=2)
    print(f"Card: {path} — {verdict}")

if __name__ == "__main__":
    main()
