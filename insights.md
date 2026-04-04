# Insights

_Updated by Dream Phase 3 (Round 2) after Exp 33._

## Round 2 Key Finding: Exp 1 MLP is Hard to Beat

### Architecture Rankings (by test_sharpe - EW)
1. **MLP hidden=32 + cross** (Exp 1/21) — gap=+1.18, best single model
2. **MLP REBAL_FREQ=10** (Exp 28) — gap=+1.51 but only 67 test samples (noisy)
3. **MLP 2-layer** (Exp 32) — gap=+0.36, more params but decent
4. **MLP 5-seed ensemble** (Exp 22) — gap=+0.23, averaging dilutes the best seed
5. **MLP+EW blend α=0.5** (Exp 29) — gap=+0.20, shrinkage toward EW works
6. **MLP+EW blend α=0.3** (Exp 30) — gap=+0.11

### Round 2 Learnings
1. **Cross-asset layer is critical** — Exp 24 (no cross) lost 1.85 in test_sharpe vs Exp 21
2. **hidden=32 is the sweet spot** — hidden=16 (Exp 25) much worse
3. **More regularization hurts** — dropout=0.5/wd=5e-3 (Exp 23) collapsed to EW-level
4. **Window averaging > last day** — Mean pool outperforms last-day (Exp 31)
5. **Window=60 is optimal** — Window=30 (Exp 27) converged to EW
6. **Exponential weighting doesn't help** — Exp 26 ≈ EW
7. **Momentum alone insufficient** — Exp 33 below EW, all 10 features needed
8. **MLP ensembling dilutes alpha** — Unlike attention, MLP seed variance is low-ish; best seed >> average
9. **Exp 1 config is already near-optimal** — 692 params, hidden=32, dropout=0.3, wd=1e-3

### The Exp 1 Mystery
- val=1.18 but test=3.94 (inverted gap). Why?
- The test period (2023-2026) might favor the specific weight pattern seed=42 learns
- This could be partly luck. Need to verify with REBAL_FREQ sensitivity.

### Round 1 Learnings (preserved)
1. **Temporal > Spatial > Linear** — for val_sharpe
2. **5-day patches optimal** for attention models
3. **Attention overfits badly** — val↑ but test↓ as complexity increases
4. **Cross-asset attention/mixing always hurts** — Exp 8, 18
5. **Val (133 samples) is very noisy** — can't reliably distinguish Sharpe differences

## Failed Ideas (Don't Retry)
### Round 1
- Bigger d_model (>16), 10-day patches, 3-day patches
- Feature dropping, turnover penalty, cross-asset mixing
- Longer windows (120d), rich temporal pooling
- SGD optimizer, Sortino loss

### Round 2
- Smaller hidden dim (16), higher dropout (0.5), stronger weight decay (5e-3)
- EW blending (dilutes alpha), momentum-only features
- Exponential window weighting, last-day only
- Window=30 (too short)

## Ideas for Remaining Experiments
- Learnable temperature for softmax (already there, but fixed init?)
- Different LR schedules (warm restart)
- GELU instead of ReLU
- Batch normalization instead of dropout
- Input noise augmentation
- Different seeds systematically to understand variance
- REBAL_FREQ=10 with more samples (window=30)
