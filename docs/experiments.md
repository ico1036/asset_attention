# Experiments

> Old experiments (0-108) archived. All had mission-drift + Sharpe bug + Y bug.
> Real start: Round 7 (Exp 109+), mission-enforced framework.

---

## Exp 0000: CrossAttentionAllocator Baseline
- **Hypothesis**: Learned asset queries cross-attend to time patches → direct temporal attention per asset → portfolio weights. Simplest possible cross-attention (217 params).
- **Architecture**: CrossAttention, d_model=8, 1 head, 12 patches, temperature=1.0
- **val_sharpe**: 0.877 | **test_sharpe**: 0.873 | **params**: 217
- **Regime signal**: entropy 2.431/2.485 (nearly uniform). Crisis-calm diff -0.026. Portfolio weights ~0.25 ± 0.002. **No regime signal.**
- **Verdict**: BASELINE — attention is uniform, model outputs near-equal weight.

## Exp 0001: Temperature Scaling (temp=0.5)
- **Hypothesis**: QK^T scores too small → softmax saturates to uniform. Halving temperature doubles logit magnitudes, sharpening attention.
- **Architecture**: Same as Exp 0000, temperature=0.5
- **val_sharpe**: 0.883 | **test_sharpe**: 0.830 | **params**: 217
- **Regime signal**: entropy 2.280 (↓ from 2.431). Crisis-calm diff -0.086 (REGIME SIGNAL). Portfolio weight std ~0.4-0.7%. Weak but present.
- **Verdict**: KEEP — temperature helps. Try more aggressive.

## Exp 0002: Aggressive Temperature (temp=0.1)
- **Hypothesis**: If temp=0.5 helps, temp=0.1 should force near-hard attention, concentrating on 1-2 key patches per asset.
- **Architecture**: Same as Exp 0000, temperature=0.1
- **val_sharpe**: 1.114 | **test_sharpe**: 0.853 | **params**: 217
- **Regime signal**: entropy 1.135 (↓↓). Crisis=1.083, Calm=1.236, diff=-0.152 (STRONG). SPY→patch9 (recent), TLT→patches 1,7 (distant). Portfolio weight std ~1.2%.
- **Key finding**: Each asset develops distinct temporal focus! SPY looks at recent data, TLT at older data. This is regime-aware behavior.
- **Verdict**: KEEP — best regime signal so far.

## Exp 0003: Learnable Temperature (init=0.1)
- **Hypothesis**: Let model learn optimal temperature. If it stays near 0.1, confirms that's the right range.
- **Architecture**: Same + nn.Parameter(log(0.1)) for temperature
- **val_sharpe**: 1.134 | **test_sharpe**: 0.850 | **params**: 218
- **Regime signal**: entropy 1.139, crisis-calm diff -0.153. Nearly identical to Exp 0002.
- **Finding**: Learned temperature converges to ~0.1. Confirms fixed temp=0.1 is sufficient.
- **Verdict**: KEEP — validates Exp 0002, no need for extra param.

## Exp 0004: Entropy Regularization in Loss
- **Hypothesis**: Even with sharp attention, portfolio weights barely vary (std ~1.2%). The score_proj+softmax smooths the attention signal. Adding entropy penalty directly to loss should force attention to be useful for differentiation.
- **Architecture**: Same as Exp 0002 (temp=0.1) + loss += 0.1 * mean_entropy
- **val_sharpe**: 1.025 | **test_sharpe**: 0.896 | **params**: 217
- **Regime signal**: entropy 0.31 (↓↓↓ very sharp). Portfolio weight std **4.4-5.1%** (4x improvement!). Crisis→calm weight shifts: SPY -2.1%, TLT +2.0%, GLD -1.4%, SHY +1.4%.
- **Key finding**: Entropy regularization is the breakthrough for portfolio weight variation. Model now meaningfully shifts allocation between regimes.
- **Verdict**: KEEP — best portfolio variation, strongest regime-dependent allocation.

---

## Summary Table (Exp 0000-0004)

| Exp | Change | val_sharpe | test_sharpe | Entropy | Crisis-Calm | Weight std |
|-----|--------|-----------|-------------|---------|-------------|------------|
| 0000 | Baseline (temp=1.0) | 0.877 | 0.873 | 2.431 | -0.026 | 0.2% |
| 0001 | temp=0.5 | 0.883 | 0.830 | 2.280 | -0.086 | 0.5% |
| 0002 | temp=0.1 | 1.114 | 0.853 | 1.135 | -0.152 | 1.2% |
| 0003 | learnable temp | 1.134 | 0.850 | 1.139 | -0.153 | 1.2% |
| 0004 | temp=0.1 + entropy reg | 1.025 | 0.896 | 0.310 | -0.081 | 4.8% |

**Key insight**: Temperature scaling (0.1) is necessary but not sufficient for portfolio variation. Entropy regularization in the loss is needed to make the attention signal actually flow through to allocation weights. The combination produces regime-dependent allocation with meaningful weight shifts.

**Next questions for Critic**:
1. Is the portfolio variation (4.8% std) meaningful enough, or still too close to equal weight?
2. Should we increase entropy reg coefficient beyond 0.1?
3. All models hover around test_sharpe ~0.85. Is attention adding value vs EW?

---

## Exp 0005-0008: Round 7 Batch 2 — Critic Verdict: REVISE

See `docs/reviews/review_r7_02.md` for detailed critique.

| Exp | Model | test_sharpe | MDD | Regime Signal | Notes |
|-----|-------|-------------|-----|---------------|-------|
| 0005 | Static (4 params) | 0.652 | -31.6% | N/A | Attention adds ~0.2 Sharpe over static |
| 0006 | CrossAttn multi-seed | 0.869±0.03 | -48.0% | Same as 0002 | Seed-robust ✓ |
| 0007 | PatchSelfAttn | 0.854 | -40.4% | ZERO | Mean pool kills signal |
| 0008 | iTransformer | 0.904 | -41.8% | Near-zero | Best Sharpe, static allocation |

**Key finding**: iTransformer architecture (asset embeddings + temporal self-attention) is best so far at 0.904 test_sharpe, but still no regime signal and still below EW (1.39).

---

## Exp 0009-0016: Round 7 Batch 3 — Attempted Critic Corrections

| Exp | Model | test_sharpe | Regime Shift | Notes |
|-----|-------|-------------|--------------|-------|
| 0009 | iTransformer + larger d_model | ~0.88 | ~0 | More capacity didn't help |
| 0010-0012 | UnifiedTemporalAttention variants | 0.47-0.49 | ~0 | Failed architecture |
| 0013 | iTransformer (re-run) | 0.931 | ~0 | Confirmed 0.93 range |
| 0014-0016 | UnifiedTemporal + diversity penalty | 0.47-0.88 | ~0 | Diversity penalty helped Sharpe but not regime |

**Key finding**: All attention models cluster around test_sharpe 0.85-0.93. EW = 1.39. Gap = 0.5 Sharpe persists. Zero regime signal in all models.

---

## Exp 0017-0021: Planned Batch 4 — Addressing Critic Feedback

**Attempted but not completed due to time constraints.**

Planned experiments to address Critic's required changes from review_r7_02:

1. **Exp 0017: iTransformer + Entropy Regularization** — Combine best architecture (iTransformer, Exp 0008) with entropy reg (Exp 0004) to boost regime signal
2. **Exp 0018: Warm-Start Training** — Carry model weights across expanding windows to help early years with small training sets
3. **Exp 0019: Minimum Training Window** — Don't start training until 5+ years of data exist (avoid tiny training sets)
4. **Exp 0020: MDD Investigation** — Print weights and returns during max drawdown period to diagnose -42% drawdown
5. **Exp 0021: EW Gap Diagnosis** — Per-year Sharpe comparison with EW to find where the 0.5 gap comes from

**Code prepared in train.py** for Exp 0017 (iTransformerEntropy) — ready to run.
