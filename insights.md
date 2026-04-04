# Insights

_Updated by Dream Phase after Exp 10._

## Patterns

### What Works
1. **Temporal attention with patching is the winner** — Exp 7 (val=2.36) is best by large margin
2. **5-day patch size is optimal** — 10-day patches (Exp 10) lost temporal granularity
3. **Last-patch pooling > mean pooling** — Most recent patch is most informative (Exp 9 vs 7)
4. **d_model=16 is the sweet spot** — d_model=12 too small (Exp 9), d_model=20+ overfits (Exp 10)
5. **Spatial attention alone is weaker** — Exp 4 (val=1.77) < Exp 7 (val=2.36)
6. **Simple mean temporal pooling is decent baseline** — MLP (Exp 1, val=1.18) beats linear (0.84)

### What Fails
1. **Adding more parameters → overfitting** — Train/val gap widens with params (Exp 5: 4.5K params, val=1.33)
2. **Dual attention is overkill** — Exp 8 (2.6K params) worse than single temporal (1.8K params)
3. **Turnover penalty hurts** — Exp 3 (val=0.76) worse than no penalty (Exp 1, val=1.18)
4. **Overlapping samples don't help with attention** — Exp 6 (val=1.59) no improvement
5. **Rich temporal pooling (last+mean+std) overfits** — Exp 2 (val=0.68) worst of all

### Key Insight
The model has only 620 training samples. Parameter efficiency is CRITICAL. The best model (Exp 7) has 1,826 params — ratio ~3.4:1. Going above 2K params consistently hurts.

## Strategy for Next 10 Experiments
1. **Keep Exp 7 as base** — PatchTemporal, 5-day, d_model=16, last-patch
2. **Try to improve without adding params**: different LR schedules, weight decay, loss functions
3. **Try 3-day patches** (20 patches) — more temporal resolution
4. **Try lightweight dual attention** — shared QKV weights between temporal and spatial
5. **Try multi-seed ensembling** — if single model plateaus
6. **Try Sortino loss or asymmetric Sharpe** — penalize downside more
7. **Do NOT try**: bigger d_model, adding FFN layers, complex architectures >2K params

## Untested Ideas
- 3-day patches
- Sortino loss
- Learnable patch mixing instead of attention
- Feature selection (drop noisy features)
- Causal masking in temporal attention
- Weight decay tuning
- Multiple random seeds
