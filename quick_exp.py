#!/usr/bin/env python3
"""Quick batch of 5 diagnostic experiments."""
import json, time, math, torch, torch.nn as nn
from prepare import load_data, make_sliding_windows, make_expanding_splits, evaluate_and_print, write_card, sharpe, N_ASSETS, MAX_PARAMS, N_PATCHES

class iTransformer(nn.Module):
    def __init__(self, n_assets=4, d_model=8, n_patches=12, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.temperature = temperature
        self.n_patches = n_patches
        self.asset_embed = nn.Embedding(n_assets, d_model)
        self.patch_embed = nn.Linear(1, d_model)
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(n_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, P, N = x.shape
        x_t = x.permute(0, 2, 1)
        h = self.patch_embed(x_t.unsqueeze(-1))
        h = h + self.pe.unsqueeze(0).unsqueeze(0)
        asset_ids = torch.arange(N, device=x.device)
        asset_emb = self.asset_embed(asset_ids)
        h = h + asset_emb.unsqueeze(0).unsqueeze(2)
        h_flat = h.reshape(B * N, P, self.d_model)
        Q = self.W_q(h_flat)
        K = self.W_k(h_flat)
        V = self.W_v(h_flat)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.temperature * math.sqrt(self.d_model))
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, V)
        pooled = context[:, -1, :]
        pooled = pooled.reshape(B, N, self.d_model)
        logits = self.score_proj(pooled).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        return weights, attn

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def train(X, Y, train_idx, val_idx, model, entropy_lambda=0.1, epochs=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    best_val, best_state, wait = -float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        w, attn = model(X_train)
        port_ret = (w * Y_train).sum(dim=-1)
        loss = -(port_ret.mean() / (port_ret.std() + 1e-8))
        row_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)
        loss = loss + entropy_lambda * row_entropy.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % 4 == 0:
            model.eval()
            with torch.no_grad():
                w_val, _ = model(X_val)
                val_ret = (w_val * Y_val).sum(dim=-1)
                vs = sharpe(val_ret)
            if vs > best_val:
                best_val, best_state, wait = vs, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 5: break
    if best_state: model.load_state_dict(best_state)
    return best_val

def run_experiment(exp_id, config, X, Y, splits, dates, tickers):
    t0 = time.time()
    results_all, yearly = [], {}
    for split in splits:
        torch.manual_seed(config['seed'])
        model = iTransformer(n_assets=config['n_assets'], d_model=config['d_model'], temperature=config['temp'])
        val_s = train(X, Y, split['train'], split['val'], model, entropy_lambda=config['entropy'])
        model.eval()
        with torch.no_grad():
            w_test, attn = model(X[split['test']])
            test_ret = (w_test * Y[split['test']]).sum(dim=-1)
            ts = sharpe(test_ret)
        year = split['test_year']
        yearly[year] = {'val_sharpe': val_s, 'test_sharpe': ts.item() if torch.is_tensor(ts) else ts, 'n_test': len(split['test'])}
        results_all.append((w_test, Y[split['test']], attn))
    
    all_w = torch.cat([r[0] for r in results_all])
    all_y = torch.cat([r[1] for r in results_all])
    ev = evaluate_and_print(all_w, all_y, 'OOS', benchmark_n=config['n_assets'])
    
    # Regime analysis on last split
    with torch.no_grad():
        _, attn_all = model(X)
    years_all = [int(dates[i][:4]) for i in range(len(X))]
    crisis_idx = [i for i, y in enumerate(years_all) if y in {2008,2009,2020,2022}]
    calm_idx = [i for i, y in enumerate(years_all) if y in {2017,2018,2019}]
    w_crisis = all_w[crisis_idx].mean(dim=0) if crisis_idx else torch.zeros(4)
    w_calm = all_w[calm_idx].mean(dim=0) if calm_idx else torch.zeros(4)
    shift = max(abs(w_crisis[i] - w_calm[i]).item() for i in range(4))
    
    regime = {'max_shift': shift, 'crisis_weights': w_crisis.tolist(), 'calm_weights': w_calm.tolist()}
    
    card = {
        'exp': exp_id,
        'config': {**config, 'model': config['name'], 'regime_signal': regime},
        'results': {
            'val_sharpe': sum(y['val_sharpe'] for y in yearly.values())/len(yearly),
            'test_sharpe': ev['sharpe'],
            'max_drawdown_pct': ev['max_drawdown_pct'],
            'turnover': ev['turnover'],
            'yearly': yearly,
            'elapsed_sec': time.time()-t0
        }
    }
    with open(f'cards/exp_{exp_id:04d}.json', 'w') as f:
        json.dump(card, f, indent=2, default=str)
    print(f"Exp {exp_id}: {config['name']} | test_sharpe={ev['sharpe']:.3f} | regime_shift={shift:.4%}")
    return ev['sharpe'], shift

def main():
    d = load_data()
    ret, dates, tickers = d['log_return'], d['dates'], d['tickers']
    X, Y, indices = make_sliding_windows(ret)
    splits = make_expanding_splits(dates, indices)
    N = len(tickers)
    
    # 5 diagnostic experiments
    experiments = [
        {'name': 'iTransformer_d4', 'd_model': 4, 'temp': 0.1, 'entropy': 0.1, 'seed': 42, 'n_assets': N},
        {'name': 'iTransformer_d16', 'd_model': 16, 'temp': 0.1, 'entropy': 0.1, 'seed': 42, 'n_assets': N},
        {'name': 'iTransformer_temp0.05', 'd_model': 8, 'temp': 0.05, 'entropy': 0.1, 'seed': 42, 'n_assets': N},
        {'name': 'iTransformer_entropy0.5', 'd_model': 8, 'temp': 0.1, 'entropy': 0.5, 'seed': 42, 'n_assets': N},
        {'name': 'iTransformer_seed0', 'd_model': 8, 'temp': 0.1, 'entropy': 0.1, 'seed': 0, 'n_assets': N},
    ]
    
    start_id = 31
    for i, cfg in enumerate(experiments):
        run_experiment(start_id + i, cfg, X, Y, splits, dates, tickers)

if __name__ == '__main__':
    main()
