# Insights

_Updated by Dream Phase 2 after Exp 20._

## Patterns

### Architecture Rankings (by best val_sharpe)
1. **3-seed ensemble of PatchTemporal** — val=2.73 (Exp 17, but test=0.22 ⚠️)
2. **3-seed ensemble of PatchTemporal** — val=2.57 (Exp 15, test=0.83)
3. **2-layer PatchTemporal shared QKV** — val=2.51 (Exp 20, test=1.06)
4. **PatchTemporal causal, single** — val=2.36 (Exp 13, test=1.44)
5. **PatchTemporal vanilla** — val=2.36 (Exp 7, test=1.44)
6. **Spatial attention only** — val=1.77 (Exp 4, test=1.33)
7. **MLP** — val=1.18 (Exp 1, test=3.94)
8. **Linear** — val=0.84 (Exp 0, test=2.56)

### Key Learnings
1. **Temporal > Spatial > Linear** — Clear hierarchy for val_sharpe
2. **5-day patches optimal** — 3-day (Exp 11) and 10-day (Exp 10) both worse
3. **Last-patch pooling > mean pooling** — Exp 9 confirmed
4. **Ensembles boost val but hurt test** — Strong sign of val overfitting
5. **Seed sensitivity is HIGH** — Val ranges from 0.71 to 2.92 across seeds
6. **Val/test divergence worsens with complexity** — Linear (test>val), attention (val>test)
7. **Cross-asset layers hurt** — Both spatial attention (Exp 8) and linear mix (Exp 18) overfit
8. **Loss function doesn't matter much** — Sortino ≈ Sharpe (Exp 12)
9. **Optimizer doesn't matter much** — SGD ≈ Adam for val (Exp 19)
10. **Causal masking is neutral** — Doesn't help or hurt (Exp 13)

### Core Problem
With only 133 val samples, val_sharpe is noisy. The equal weight benchmark (2.76) may be hard to
beat robustly because:
- Equal weight is diversification-optimal for uncorrelated assets
- Any learned allocation concentrates, increasing variance
- 133 samples is not enough to reliably estimate Sharpe differences

### Best Honest Single Model
Exp 13 (PatchTemporal, causal, val=2.36) is the best **single model** — reproducible,
no seed-picking, reasonable val/test ratio.

## Failed Ideas (Don't Retry)
- Bigger d_model (>16) → overfits
- 10-day patches → loses granularity
- 3-day patches → too noisy
- Feature dropping → hurts
- Turnover penalty → hurts
- Cross-asset mixing → hurts
- Longer windows (120d) → hurts
- Rich temporal pooling (last+mean+std) → hurts
- SGD optimizer → no improvement

## Untested Ideas for Future
- RoPE position encoding (instead of learned)
- SwiGLU activation in FFN
- Learnable patch aggregation (weighted avg not fixed last)
- Pre-training on broader stock universe
- Rolling window validation (Phase 2 from philosophy.md)
- Data augmentation (noise injection, temporal jitter)
