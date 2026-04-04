# Insights

_Updated by Dream Phase (Round 3 final) after Exp 55._

## The Definitive Answer: Attention Does NOT Help for This Task

After 55 experiments across 3 rounds, the conclusion is clear:
**Self-attention (spatial, temporal, or dual) does not improve over a simple MLP for 17-ETF allocation with ~620 clean samples.**

## Best Model
**GELU MLP + noise=0.1** (Exp 36/46): val≈1.85, test≈4.2, 692 params
- Architecture: mean-pool → Linear(10→32) → GELU → Dropout(0.3) → Linear(32→1) → Linear(17→17 cross) → softmax
- The 17×17 cross-asset linear layer is critical (Exp 24: removing it drops test from 3.94 to 2.08)
- Gaussian noise (σ=0.1) during training acts as implicit regularization

## But Is the Alpha Real? NO.
**Exp 55 revealed the truth**: Seed 42's test_sharpe=4.16 comes from:
- **40.6% SHY** (short-term treasuries — essentially cash)
- **12.5% UUP** (US dollar index)
- Low equity exposure

This is a **defensive/cash-heavy portfolio** that happens to excel in the 2020-2026 test period (rate hikes, equity drawdowns, dollar strength). It is NOT a learned cross-asset regime-switching strategy.

Evidence:
- Different split (60/20/20): test drops from 4.16 to 2.47 (barely above EW=2.27)
- Other seeds that find similar SHY-heavy allocations (seed 256: 54% SHY) also beat EW
- Seeds that learn near-uniform weights (7, 99) get exactly EW returns

## Round 3 Discoveries

### 1. Daily Rebalancing (REBAL_FREQ=1) Is Harmful
- Single-day return targets are too noisy for 60-day feature windows
- EW Sharpe drops from 2.76 → 1.36
- All models perform worse with daily targets

### 2. Overlapping Samples (stride=1) Are Harmful
- Creates autocorrelated targets (4/5 days shared between consecutive samples)
- Model overfits autocorrelation (train_sharpe=7.2) but val/test degrade
- **Stride must equal target horizon for clean, independent samples**

### 3. "Data Scarcity" Was a Red Herring
- 620 samples is sufficient for a 692-param MLP
- The problem was never sample count — it's that the task is inherently hard
- No data augmentation trick (overlap, noise, bootstrap) changes the fundamental conclusion

### 4. Attention Consistently Fails (Now With Noise)
Even with noise augmentation (the best regularizer from Round 2):
- Spatial attention: test=0.83 (EW=2.76) ❌
- PatchTemporal: test=2.38 (EW=2.76) ❌
- Dual attention: test=1.42 (EW=2.76) ❌
- MLP-Mixer: test=1.34 (EW=2.76) ❌
- Attention cross-mixing: test=1.98 (EW=2.76) ❌

### 5. SWA Hurts This Task
Stochastic Weight Averaging averages toward a worse minimum (test: 4.19→1.82).

## Why Attention Fails Here
1. **Too few independent samples** (620) for attention's O(n²) parameters to learn
2. **Mean-pooling destroys temporal structure** that attention needs — but the MLP's mean-pool is optimal
3. **The cross-asset linear layer captures what spatial attention tries to learn**, but with 289 params vs 1000+
4. **The signal is simple**: overweight safe assets in volatile times. A linear cross layer suffices.

## Failed Ideas (Complete List)
### Data Pipeline
- REBAL_FREQ=1, stride=1 overlap, stride=1 with non-overlap eval, SWA

### Architectures (all fail to beat MLP on test)
- Spatial attention, PatchTemporal, Dual attention, MLP-Mixer, Attention cross-mixing
- 2-layer MLP, hidden≠32, no cross layer, LayerNorm, attention + FFN blocks

### Training
- SGD, Sortino loss, turnover penalty, EW blending, SWA, higher dropout/WD

## Final Assessment
The project answered its core question: **For small-scale ETF allocation (17 assets, 620 samples), attention mechanisms add complexity without benefit.** A simple 692-param MLP with noise augmentation and a cross-asset linear layer is optimal.

The apparent "alpha" (test_sharpe=4.2) is a SHY-overweight strategy that works in the specific test period, not a robust learned strategy. A 60/20/20 split reduces the edge to near-zero.
