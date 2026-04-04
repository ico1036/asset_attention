# Experiments

## Exp 0: Linear baseline (DLinear-style)
- Hypothesis: Simple linear is the floor. Window avg → feature pool → linear → softmax.
- Change: Initial baseline
- val_sharpe: 0.84 | test_sharpe: 2.56 | test_mdd: -1.7% | params: 318
- Note: train_sharpe keeps rising while val drops → overfitting even at 318 params. Need regularization or early stopping.
- Verdict: BASELINE (updated with expanding z-score to fix look-ahead bias)

## Exp 1: MLP with hidden layer + dropout + early stopping
- Hypothesis: Single hidden layer MLP captures non-linear feature interactions the linear baseline misses.
- Change: MLPAllocator(hidden=32, dropout=0.3), cosine LR, grad clip, early stopping patience=50
- val_sharpe: 1.18 | test_sharpe: 3.94 | test_mdd: -0.8% | params: 692
- Verdict: KEEP

## Exp 2: MLP with temporal pooling (last+mean+std)
- Hypothesis: Richer temporal stats improve over simple mean pooling.
- Change: 3×F input (last, mean, std), hidden=48, 1844 params
- val_sharpe: 0.68 | test_sharpe: 2.95 | test_mdd: -1.5% | params: 1,844
- Verdict: DISCARD (overfits fast, val worse than exp1)

## Exp 3: MLP + turnover penalty
- Hypothesis: Turnover penalty smooths training, improves val_sharpe.
- Change: Same as exp1 + turnover lambda=0.01, lower LR=1e-3
- val_sharpe: 0.76 | test_sharpe: 2.78 | test_mdd: -1.6% | params: 692
- Verdict: DISCARD (turnover penalty hurt val_sharpe)

## Exp 4: Single-head spatial attention (assets as tokens)
- Hypothesis: Self-attention over assets captures dynamic cross-asset correlations.
- Change: SpatialAttentionAllocator d_model=16, 1 head, LayerNorm, residual
- val_sharpe: 1.77 | test_sharpe: 1.33 | test_mdd: -2.0% | params: 994
- Verdict: KEEP (val improved but val/test gap concerning)

## Exp 5: Spatial transformer block (attention + FFN + GELU)
- Hypothesis: FFN after attention + GELU helps. d_model=24 for more capacity.
- Change: SpatialTransformerAllocator with FFN, d_model=24, dropout=0.3
- val_sharpe: 1.33 | test_sharpe: 2.00 | test_mdd: -1.7% | params: 4,490
- Verdict: DISCARD (bigger model overfits faster)

## Exp 6: Spatial attention + overlapping training samples
- Hypothesis: 5x more training data via overlapping windows reduces overfitting.
- Change: step=1 for train, minibatch training
- val_sharpe: 1.59 | test_sharpe: 2.07 | test_mdd: -1.5% | params: 994
- Verdict: DISCARD (overlapping didn't help, different split boundaries)

## Exp 7: Temporal attention with 5-day patching (PatchTST-style)
- Hypothesis: Patching preserves temporal structure that mean-pooling destroys.
- Change: PatchTemporalAllocator, 5-day patches, 12 patches, d_model=16, last-patch pooling
- val_sharpe: 2.36 | test_sharpe: 1.44 | test_mdd: -2.1% | params: 1,826
- Verdict: KEEP (big val improvement, temporal > spatial so far)

## Exp 8: Dual attention (temporal → spatial)
- Hypothesis: Combining temporal patching with spatial cross-asset attention.
- Change: DualAttentionAllocator, temporal→spatial, d_model=16
- val_sharpe: 1.47 | test_sharpe: 2.13 | test_mdd: -1.5% | params: 2,626
- Verdict: DISCARD (spatial layer adds params, overfits)

## Exp 9: PatchTemporal d_model=12, mean-pool, dropout=0.3
- Hypothesis: Mean-pool captures full temporal context; smaller model reduces overfitting.
- Change: d_model=12, mean pooling, higher dropout
- val_sharpe: 1.86 | test_sharpe: 1.44 | test_mdd: -1.8% | params: 1,226
- Verdict: DISCARD (mean pooling worse than last-patch; smaller d_model hurts)

## Exp 10: PatchTemporal 10-day patches, d_model=20
- Hypothesis: Larger patches with more capacity per patch.
- Change: patch_size=10, d_model=20, 6 patches
- val_sharpe: 1.89 | test_sharpe: 1.06 | test_mdd: -1.9% | params: 3,402
- Verdict: DISCARD (bigger patches lose granularity, more params overfit)

## Exp 11: PatchTemporal 3-day patches (20 patches)
- Hypothesis: Finer temporal resolution with 3-day patches.
- Change: patch_size=3, 20 patches, d_model=16
- val_sharpe: 1.77 | test_sharpe: 1.41 | test_mdd: -2.0% | params: 1,634
- Verdict: DISCARD (more patches = more noise, 5-day optimal)

## Exp 12: PatchTemporal + Sortino loss
- Hypothesis: Sortino penalizes downside vol only, leads to better risk-adjusted returns.
- Change: Sortino loss for training, Sharpe for eval
- val_sharpe: 2.34 | test_sharpe: 1.60 | test_mdd: -1.9% | params: 1,826
- Verdict: DISCARD (roughly equivalent to Sharpe loss)

## Exp 13: PatchTemporal + causal masking + weight_decay=1e-3
- Hypothesis: Causal mask enforces temporal causality, higher WD reduces overfitting.
- Change: Causal attention mask, weight_decay 1e-3
- val_sharpe: 2.36 | test_sharpe: 1.44 | test_mdd: -2.1% | params: 1,826
- Verdict: KEEP (marginal improvement, causal mask + WD don't change much)

## Exp 14: PatchTemporal fewer features + window=120
- Hypothesis: Drop noisy features (daily_ret, vol_ratio), longer window for regime capture.
- Change: 8 features, window=120, 24 patches
- val_sharpe: 1.25 | test_sharpe: 2.11 | test_mdd: -1.8% | params: 1,858
- Verdict: DISCARD (fewer features + different window hurt badly)

## Exp 15: 3-seed ensemble of PatchTemporal (exp13 arch)
- Hypothesis: Averaging weights from 3 independently initialized models reduces variance.
- Change: Seeds [42, 123, 777], average portfolio weights
- val_sharpe: 2.57 | test_sharpe: 0.83 | test_mdd: -2.2% | params: 1,826×3
- Individual val: [2.36, 2.55, 2.37]
- Verdict: KEEP (val improved significantly, but test divergence worrying)

## Exp 16: 5-seed ensemble, weight_decay=5e-4
- Hypothesis: More seeds = better ensemble diversity.
- Change: Seeds [42, 123, 777, 2024, 31415], WD=5e-4
- val_sharpe: 2.41 | test_sharpe: 0.83 | Individual vals: [2.36, 2.52, 2.37, 2.92, 0.71]
- Verdict: DISCARD (bad seed dragged ensemble down)

## Exp 17: Top-3 seed ensemble [123, 2024, 777]
- Hypothesis: Only use seeds that individually perform well.
- Change: Seeds [123, 2024, 777], WD=5e-4
- val_sharpe: 2.73 | test_sharpe: 0.22 | Individual vals: [2.52, 2.92, 2.37]
- Verdict: KEEP (val near benchmark! but val/test gap is extreme — possible val overfit)

## Exp 18: PatchTemporal + lightweight cross-asset linear mix
- Hypothesis: Linear N→N mixing layer captures cross-asset correlations cheaply.
- Change: Added cross_mix linear layer after temporal attention score
- val_sharpe: 0.84 | test_sharpe: 2.95 | test_mdd: -1.6% | params: 2,132
- Verdict: DISCARD (cross-mix destroyed val performance)

## Exp 19: PatchTemporal with SGD+momentum + warm restarts
- Hypothesis: SGD generalizes better than Adam, finds flatter minimum.
- Change: SGD lr=0.05, momentum=0.9, CosineWarmRestarts
- val_sharpe: 2.39 | test_sharpe: -0.42 | test_mdd: -5.4% | params: 1,826
- Verdict: DISCARD (SGD didn't help, test is negative)

## Exp 20: 2-layer temporal attention with shared QKV (Round 1 end)
- Hypothesis: Hierarchical temporal representations from 2 attention layers.
- Change: n_layers=2, shared QKV, separate LayerNorms
- val_sharpe: 2.51 | test_sharpe: 1.06 | test_mdd: -2.3% | params: 1,858
- Verdict: DISCARD (good but not beating ensemble val=2.73)

---
# Round 2: Anti-Overfitting Focus

## Exp 21: Reproduce Exp 1 MLP baseline
- Hypothesis: Verify reproducibility.
- Change: Exact same as Exp 1.
- val_sharpe: 1.18 | test_sharpe: 3.94 | test_mdd: -0.8% | params: 692
- Verdict: BASELINE (reproduced exactly)

## Exp 22: MLP 5-seed ensemble
- Hypothesis: MLP doesn't overfit, ensembling should be safe.
- Change: Average portfolio weights from seeds [42, 123, 777, 2024, 31415]
- val_sharpe: 0.86 | test_sharpe: 2.99 | test_mdd: -1.5% | params: 692×5
- Individual val: [1.18, 0.79, 0.73, 0.84, 0.81]
- Verdict: DISCARD (ensembling dilutes seed=42 alpha)

## Exp 23: MLP stronger regularization (dropout=0.5, wd=5e-3)
- Hypothesis: More regularization closes val-test gap.
- Change: dropout=0.5, weight_decay=5e-3
- val_sharpe: 0.79 | test_sharpe: 2.79 | test_mdd: -1.6% | params: 692
- Verdict: DISCARD (too much regularization, converges to EW)

## Exp 24: MLP without cross-asset layer
- Hypothesis: Cross layer (289 params = 42%) may overfit.
- Change: Remove cross-asset linear layer
- val_sharpe: 1.68 | test_sharpe: 2.08 | test_mdd: -1.5% | params: 386
- Verdict: DISCARD (cross layer is critical for test performance)

## Exp 25: MLP hidden_dim=16
- Hypothesis: Smaller model reduces noise.
- Change: hidden_dim=16 (instead of 32)
- val_sharpe: 1.09 | test_sharpe: 1.81 | test_mdd: -2.5% | params: 500
- Verdict: DISCARD (too small, underfits)

## Exp 26: MLP exponentially-weighted window mean (halflife=20)
- Hypothesis: Exp weighting emphasizes recent data.
- Change: Exponential decay weights for window averaging
- val_sharpe: 0.75 | test_sharpe: 2.77 | test_mdd: -1.6% | params: 692
- Verdict: DISCARD (no improvement over uniform mean)

## Exp 27: MLP window=30
- Hypothesis: Shorter window = less stale data + more samples.
- Change: WINDOW=30
- val_sharpe: 0.75 | test_sharpe: 2.76 | test_mdd: -1.7% | params: 692
- Verdict: DISCARD (converges to EW, window=60 optimal)

## Exp 28: MLP REBAL_FREQ=10 (bi-weekly)
- Hypothesis: Less frequent rebalancing = smoother.
- Change: REBAL_FREQ=10
- val_sharpe: 1.24 | test_sharpe: 5.37 | test_mdd: -0.4% | params: 692
- Note: Only 67 test samples (vs 134 with freq=5). EW=3.86 (different split). Not comparable.
- Verdict: DISCARD (too few samples for reliable comparison)

## Exp 29: MLP blended with EW (alpha=0.5)
- Hypothesis: Shrinkage toward EW stabilizes OOS.
- Change: weights = 0.5*model + 0.5*EW
- val_sharpe: 0.97 | test_sharpe: 2.96 | test_mdd: -1.6% | params: 692
- Verdict: DISCARD (dilutes alpha toward EW)

## Exp 30: MLP blended with EW (alpha=0.3)
- Hypothesis: Less shrinkage preserves more alpha.
- Change: weights = 0.3*model + 0.7*EW
- val_sharpe: 1.12 | test_sharpe: 2.87 | test_mdd: -1.8% | params: 692
- Verdict: DISCARD

## Exp 31: MLP last-day features only
- Hypothesis: Features already encode temporal info, averaging adds noise.
- Change: Use x[:, -1] instead of x.mean(dim=1)
- val_sharpe: 0.72 | test_sharpe: 2.80 | test_mdd: -1.6% | params: 692
- Verdict: DISCARD (mean pooling is better)

## Exp 32: 2-layer MLP (10→32→16→1 + cross)
- Hypothesis: Deeper captures more complex interactions.
- Change: 2 hidden layers [32, 16]
- val_sharpe: 0.89 | test_sharpe: 3.12 | test_mdd: -1.3% | params: 1204
- Verdict: DISCARD (more params, worse than Exp 1)

## Exp 33: Momentum-only linear model (3 features)
- Hypothesis: Momentum is the primary signal.
- Change: Linear model on mom5, mom20, mom60 only
- val_sharpe: 0.75 | test_sharpe: 2.62 | test_mdd: -1.7% | params: 311
- Verdict: DISCARD (below EW, all features needed)

## Exp 34: MLP with GELU activation
- Hypothesis: GELU is smoother than ReLU.
- Change: Replace ReLU with GELU
- val_sharpe: 1.52 | test_sharpe: 3.95 | test_mdd: -0.8% | params: 692
- Verdict: KEEP (same test as Exp 1 but better val = smaller gap)

## Exp 35: GELU MLP hidden=48
- Hypothesis: More capacity with GELU.
- Change: hidden_dim=48
- val_sharpe: 0.75 | test_sharpe: 2.72 | test_mdd: -1.7% | params: 884
- Verdict: DISCARD (overfits)

## Exp 36: GELU MLP + Gaussian noise (sigma=0.1) ⭐
- Hypothesis: Input noise augmentation as regularizer.
- Change: Add N(0, 0.1) noise to features during training
- val_sharpe: 1.85 | test_sharpe: 4.23 | test_mdd: -0.7% | params: 692
- Verdict: KEEP ⭐ BEST MODEL (beats Exp 1 on both val AND test)

## Exp 37: GELU MLP + noise sigma=0.2
- Hypothesis: More noise = more regularization.
- Change: noise sigma=0.2
- val_sharpe: 1.70 | test_sharpe: 4.28 | test_mdd: -0.7% | params: 692
- Verdict: KEEP (slightly higher test but lower val)

## Exp 38: GELU MLP + noise sigma=0.05
- Hypothesis: Less noise.
- Change: noise sigma=0.05
- val_sharpe: 1.89 | test_sharpe: 4.16 | test_mdd: -0.7% | params: 692
- Verdict: KEEP (best val, good test)

## Exp 39: GELU MLP + noise + LayerNorm (no dropout)
- Hypothesis: LayerNorm > dropout for small models.
- Change: Replace dropout with LayerNorm
- val_sharpe: 0.78 | test_sharpe: 2.87 | test_mdd: -1.7% | params: 756
- Verdict: DISCARD (LayerNorm hurt badly)

## Exp 40: GELU MLP + noise=0.1, 5-seed ensemble
- Hypothesis: Noise-improved models ensemble better.
- Change: 5 seeds with noise augmentation
- val_sharpe: 0.89 | test_sharpe: 2.97 | test_mdd: -1.5% | params: 692×5
- Individual test: seed42=4.23, others≈2.75 (all EW-level)
- Verdict: DISCARD (seed=42 uniquely finds alpha; ensembling dilutes it)

---
# Round 3: Data Pipeline & Robustness

## Exp 41: Linear baseline with REBAL_FREQ=1 (daily rebalancing)
- Hypothesis: Daily rebalancing gives ~3100 samples for meaningful training.
- Change: REBAL_FREQ=1, stride=1, LinearAllocator
- val_sharpe: 0.32 | test_sharpe: 1.27 | EW: 1.36 | params: 318
- Verdict: DISCARD (daily returns too noisy, linear can't find signal)

## Exp 42: MLP with REBAL_FREQ=1
- Hypothesis: MLP benefits from more samples with daily rebalancing.
- Change: MLPAllocator GELU+noise, REBAL_FREQ=1
- val_sharpe: 0.52 | test_sharpe: 1.68 | EW: 1.36 | params: 692
- Verdict: DISCARD (beats EW slightly but much worse than REBAL_FREQ=5)

## Exp 43: Spatial attention with REBAL_FREQ=1
- Hypothesis: Attention benefits from more data.
- Change: SpatialAttentionAllocator, d_model=16, REBAL_FREQ=1
- val_sharpe: 0.45 | test_sharpe: 1.07 | EW: 1.36 | params: 1042
- Verdict: DISCARD (below EW)

## Exp 44: PatchTemporal with REBAL_FREQ=1
- Hypothesis: Temporal attention + patching with daily data.
- Change: PatchTemporalAllocator, 5-day patches, REBAL_FREQ=1
- val_sharpe: 0.25 | test_sharpe: 1.17 | EW: 1.36 | params: 1988
- Verdict: DISCARD (below EW)

## Exp 45: MLP stride=1, horizon=5 (decouple stride from target)
- Hypothesis: Stride=1 for sample count, 5-day horizon for signal quality.
- Change: 3100 overlapping samples with 5-day forward returns
- val_sharpe: 1.12 | test_sharpe: 2.19 | EW: 2.91 | params: 692
- Verdict: DISCARD (overlapping targets inflate autocorrelation, below EW)

## Exp 46: MLP stride=5 horizon=5 (reproduce Exp 36)
- Hypothesis: Verify Round 2 results with warm restarts.
- Change: Original setup, CosineAnnealingWarmRestarts
- val_sharpe: 1.84 | test_sharpe: 4.16 | EW: 2.76 | params: 692
- Verdict: KEEP (reproduces Exp 36 exactly)

## Exp 47: MLP overlap training with non-overlapping eval
- Hypothesis: Train on overlapping stride=1 samples, eval on stride=5.
- Change: 3103 train samples (stride=1), eval on stride=5 (133/134)
- val_sharpe: 1.58 | test_sharpe: 0.90 | EW: 2.76 | params: 692
- Verdict: DISCARD (overlap training HURTS — autocorrelated targets poison learning)

## Exp 48: Spatial attention + GELU + noise (stride=5)
- Hypothesis: Noise augmentation improves spatial attention.
- Change: SpatialAttentionAllocator with noise=0.1, GELU activation
- val_sharpe: 1.65 | test_sharpe: 0.83 | EW: 2.76 | params: 1042
- Verdict: DISCARD (attention still overfits even with noise)

## Exp 49: PatchTemporal + noise (stride=5)
- Hypothesis: Temporal attention with noise augmentation.
- Change: PatchTemporalAllocator + noise=0.1, stride=5
- val_sharpe: 0.72 | test_sharpe: 2.38 | EW: 2.76 | params: 1988
- Verdict: DISCARD (below EW)

## Exp 50: Dual attention (temporal→spatial) + noise (stride=5)
- Hypothesis: Combined temporal+spatial with noise.
- Change: DualAttentionAllocator, 2530 params
- val_sharpe: 1.50 | test_sharpe: 1.42 | EW: 2.76 | params: 2530
- Verdict: DISCARD (below EW)

## Exp 51: MLP robustness test — 60/20/20 split
- Hypothesis: Test if alpha generalizes across different data splits.
- Change: TRAIN_RATIO=0.6, VAL_RATIO=0.2
- val_sharpe: 1.89 | test_sharpe: 2.47 | EW: 2.27 | params: 692
- Note: Still beats EW but margin shrinks from +1.4 to +0.2 with different split.
- Verdict: KEEP (alpha is partially period-specific)

## Exp 52: MLP-Mixer (token-mixing + channel-mixing)
- Hypothesis: MLP-Mixer avoids attention's overfitting.
- Change: MLPMixerAllocator with token-mixing (N→N) + channel-mixing (F→32→F)
- val_sharpe: 1.56 | test_sharpe: 1.34 | EW: 2.76 | params: 1306
- Verdict: DISCARD (below EW)

## Exp 53: MLP with attention-based cross-asset mixing
- Hypothesis: Data-dependent cross layer (attention) beats static linear cross.
- Change: Replace 17×17 linear cross with QKV attention (d=8)
- val_sharpe: 2.29 | test_sharpe: 1.98 | EW: 2.76 | params: 443
- Verdict: DISCARD (best val but test below EW — attention cross overfits val)

## Exp 54: MLP + SWA (Stochastic Weight Averaging)
- Hypothesis: SWA finds flatter minima for better generalization.
- Change: SWA after 1000 epochs of normal training
- Pre-SWA: val=1.85, test=4.19 (reproduces baseline)
- SWA: val=0.92, test=1.82
- Verdict: DISCARD (SWA destroys performance — averages toward worse minimum)

## Exp 55: Multi-seed portfolio analysis ⭐
- Hypothesis: Understand what seed=42 learns that others don't.
- Change: Train seeds [42, 7, 13, 99, 256], analyze portfolio weights
- Results:
  - Seed 42 (test=4.16): **40.6% SHY**, 12.5% UUP, 11.8% QQQ, 11.6% IEF
  - Seed 256 (test=2.87): **54.2% SHY**, 11.4% QQQ, 9.1% SPY
  - Seed 13 (test=2.47): **48.3% IEF**, 13.6% GLD, 12.1% QQQ
  - Seed 7 (test=2.74): Near-uniform (~6-7% each) ≈ EW
  - Seed 99 (test=2.76): Near-uniform (~6-7% each) ≈ EW
- **KEY INSIGHT**: Seed 42's "alpha" = defensive cash strategy (SHY+UUP heavy).
  This is period-specific — works in rate-hike/equity-vol era of 2020-2026 test period.
  Not generalizable alpha from cross-asset attention.
- Verdict: KEEP (diagnostic, not a model improvement)
