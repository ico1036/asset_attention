# Insights

_Updated by Dream Phase (Round 4 final) after Exp 69. 69 experiments across 4 rounds._

## The Definitive Answer (Revised from Round 3)

**Round 1-3 concluded "attention doesn't help" — this was WRONG due to broken training.**
**Round 4 corrected answer: Attention works as well as MLP; NEITHER beats Equal Weight.**

## What Went Wrong in Rounds 1-3

All models (MLP and attention alike) trained in 1-2 seconds via full-batch gradient descent on 620 samples. This found random sharp minima that were:
1. **Seed-dependent**: Seed 42 found a SHY-heavy allocation; other seeds found ~EW
2. **Not generalizable**: The "alpha" was a cash-overweight strategy specific to 2020-2026
3. **Misleading for comparison**: Full-batch favored the simpler MLP (fewer params = faster convergence to a specific minimum), making attention look worse

## Round 4 Proper Training Results

With mini-batch training (batch_size=64, LR=5e-4), which adds gradient noise as implicit regularization:

### Final 3-Way Comparison (Exp 69, 5 seeds each):
| Model | Params | Val Mean | Test Mean | Test Median |
|-------|--------|----------|-----------|-------------|
| Linear | 318 | 1.11 | 1.73 | 1.57 |
| MLP | 692 | 0.92 | 2.42 | 2.57 |
| **Spatial Attn+Cross** | 944 | **1.48** | 2.17 | **2.57** |
| Equal Weight | 0 | — | **2.76** | **2.76** |

Key findings:
- **Attention has the highest val_sharpe** (1.48 vs MLP 0.92) — it generalizes better to unseen validation data
- **Test medians are identical** for MLP and Attention (both 2.57) — no statistical difference
- **Equal Weight beats all models on test** — no model architecture finds reliable alpha
- Attention is **more consistent** across seeds (lower variance)

### Complexity Ladder (properly trained):
- Linear (318p): val=1.11, test=1.73 — underfit
- MLP (692p): val=0.92, test=2.42 — high test variance, seed-dependent
- Spatial Attn (1042p): val=1.19, test=1.97 — consistent but below EW
- **Spatial Attn+Cross (944p): val=1.48, test=2.17** — best val, decent test
- PatchTemporal (1988p): val=0.72, test=2.42 — too many params, underfits val
- Dual Attention (1754p): val=1.90, test=-0.70 — catastrophic overfitting

## Why No Model Beats EW

1. **620 independent samples** (weekly rebalancing × 12 years) is fundamentally insufficient
2. The **signal-to-noise ratio** in multi-asset allocation is very low
3. With enough regularization, all models converge toward EW-like allocations
4. Without enough regularization, all models overfit (train_sharpe >> val_sharpe)
5. The "alpha" region between EW-convergence and overfitting is too narrow and seed-dependent

## What We Learned About Attention

1. **Attention is NOT worse than MLP** — with proper training it's actually better on val
2. **Spatial attention captures meaningful cross-asset structure** — higher val than linear
3. The **linear cross layer** (17×17 = 289 params) is a highly efficient proxy for spatial attention
4. **Temporal attention (PatchTST) adds too many params** for 620 samples
5. **Dual attention catastrophically overfits** — too complex for this data regime

## Confirmed Findings (unchanged from earlier rounds)
- Daily rebalancing (REBAL_FREQ=1) is harmful — single-day returns too noisy
- Overlapping samples (stride=1) poison learning via autocorrelated targets
- Stride must equal target horizon for clean samples
- SWA hurts — averages toward worse minimum
- GELU > ReLU for this task
- Mean-pooling over time is optimal (attention over time doesn't help)

## Recommendations
1. **Use EW** as the production allocation for this 17-ETF universe
2. If a model is desired: Spatial Attention + Cross layer (best val generalization)
3. For more data: expand the ETF universe or use shorter rebalancing with different feature design
4. The real opportunity may be in **feature engineering**, not model architecture
