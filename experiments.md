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
