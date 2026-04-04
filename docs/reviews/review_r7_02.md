# CIO Review: Round 7 Batch 2, Experiments 0005–0008

**Reviewer**: Critic Agent (Quant CIO)  
**Date**: 2026-04-04  
**Scope**: Static ablation, multi-seed robustness, PatchSelfAttention, iTransformer  
**Prior review**: review_r7_01.md (Verdict: REVISE)

---

## Verdict: REVISE

---

## Did the Explorer Address My Required Changes?

| Required Change | Status |
|----------------|--------|
| Static-weight ablation | ✅ Done (Exp 0005) |
| Multi-seed testing | ✅ Done (Exp 0006) |
| Explain -44% MDD | ❌ Not addressed |
| Compare to EW honestly | ❌ Not addressed — gap widened |
| Drop "breakthrough" framing | ✅ Framing is neutral now |

Credit where due: the Explorer executed the ablation and multi-seed tests I asked for. Good discipline. But the two hardest questions — MDD and EW underperformance — were dodged.

---

## Experiment-by-Experiment

### Exp 0005: Static Weight Ablation — THE KEY RESULT

**test_sharpe: 0.652** (attention models: 0.85–0.90)

This is the most important experiment in the batch. A model with **4 learnable parameters and no attention** gets test_sharpe 0.652. The attention models get ~0.87. That means attention adds **~0.2 Sharpe** over static weights.

Is 0.2 Sharpe meaningful? Marginally. It proves attention isn't completely inert. But:

- EW (zero parameters) gets **1.39**. Static learned weights get 0.652. Attention gets 0.87. All three underperform EW.
- The static model reveals something fascinating in `learned_weights_by_year`: some years it goes 94% SHY (2016, 2019-2021, 2025). Other years it's near equal-weight. **The static model is doing regime switching through retraining** — each expanding window produces different optimal static weights.
- MDD: -31.6% for static vs -41 to -48% for attention models. **The static model has better drawdown control.** That's damning.

**My take**: Attention adds a small positive signal (~0.2 Sharpe) but also adds instability (worse MDD). Net value: questionable.

### Exp 0006: Multi-Seed CrossAttention

**Mean test_sharpe: 0.869 ± 0.030**

Good news: very tight std (0.03). The model is seed-robust. This was a real concern from R1-6, and it's resolved. Five seeds all land in [0.845, 0.926].

Bad news: the MDD is -48%. Every seed produces roughly the same mediocre result. It's consistently mediocre.

### Exp 0007: PatchSelfAttention

**test_sharpe: 0.854, max_shift: 0.00001, weight std: ~0.0001**

This model is dead on arrival. Per-asset self-attention with mean pooling → score → softmax. The regime signal is **zero**. Crisis-calm entropy diff: -0.00006. Portfolio weights: 25.00% ± 0.00%. It's equal weight with extra computation.

The architecture flaw is clear: mean pooling over the temporal dimension **after** self-attention destroys all the information attention learned. The attention is there, but its output gets averaged away. This is exactly the antipattern philosophy.md warns about.

### Exp 0008: iTransformer (Best of Batch)

**test_sharpe: 0.904, val_sharpe: 1.047**

Best test Sharpe so far. The architecture is interesting: asset embeddings + temporal self-attention per asset + last-patch pooling (not mean). Several things I like:

- **Asset embeddings give identity**: each asset has a learned embedding that differentiates its temporal attention. This is why iTransformer works better than PatchSelfAttention.
- **Last-patch pooling** instead of mean — preserves recency signal.
- **Val-test gap is reasonable** (1.05 → 0.90). Not wildly overfit.

But the regime signal is still near-zero. Crisis weights: [33.4%, 20.0%, 26.0%, 20.6%]. Calm weights: identical to 4 decimal places. Max shift: 0.00003. **The model found better static weights (overweight SPY, underweight TLT/SHY) but doesn't shift between regimes.**

The 0.904 comes from a better fixed allocation, not from regime-aware dynamics. That's an improvement over EW-mimicry, but it's not the mission.

---

## The Elephant in the Room: EW = 1.39

All attention models cluster around 0.85–0.90. EW gets 1.39. **The gap is 0.5 Sharpe.** This has persisted through 9 experiments. Nobody has explained why.

Let me hypothesize: the expanding-window retraining is the culprit. Each year, the model is retrained from scratch on all prior data. The early years have tiny training sets. Models trained on 2-3 years of data are bad. Their test Sharpes drag down the aggregate. EW doesn't have this problem — it's always 25/25/25/25.

Look at Exp 0005's yearly results: test Sharpes of -0.74, -0.28, +3.99 by year. The variance is enormous. EW's consistency wins over the model's instability.

**This is not an attention problem. It's a training protocol problem.** The expanding window with small initial sets is sabotaging all learned models. The Explorer should investigate this.

---

## Code Review (train.py — Exp 0008)

The code is clean and well-commented. The data journey comment at the top is excellent. A few notes:

1. **Seed reset per split** (`torch.manual_seed(42)` in the loop) — good, reproducible.
2. **Attention analysis uses only the last model** — same issue as before. The regime analysis reflects training on all data through 2025, not a consistent model.
3. **The model actually does temporal self-attention per asset** (not spatial as iTransformer paper does). It's more like PatchTST with asset embeddings. The "iTransformer" name is misleading — the original paper does attention across variables, not within time per variable. The Explorer's own comment in the code noticed this ("Wait — this violates attention over time dimension") and pivoted, which is good thinking.
4. **249 params, well under 25K limit.** No issue.

---

## Summary Table (All Experiments)

| Exp | Model | test_sharpe | MDD | Regime Signal | Notes |
|-----|-------|-------------|-----|---------------|-------|
| 0005 | Static (4 params) | 0.652 | -31.6% | N/A (ablation) | Baseline for attention value |
| 0006 | CrossAttn (multi-seed) | 0.869±0.03 | -48.0% | Same as 0002 | Seed-robust ✓ |
| 0007 | PatchSelfAttn | 0.854 | -40.4% | ZERO | Mean pool kills signal |
| 0008 | iTransformer | 0.904 | -41.8% | Near-zero | Best Sharpe, static allocation |
| — | EW | 1.39 | ? | N/A | Still king |

---

## What's Good

1. **The ablation answered a real question.** Attention adds ~0.2 Sharpe over static weights. Small but real.
2. **Multi-seed test shows robustness.** σ=0.03 across 5 seeds. The model is stable.
3. **iTransformer is the right direction.** Asset embeddings + temporal attention per asset is a better architecture than cross-attention with learned queries. 0.904 > 0.87.
4. **Experimental discipline continues to be strong.** One change per experiment, clear hypotheses.

## What's Wrong

1. **EW gap unexplained.** 0.5 Sharpe deficit after 9 experiments. This is the #1 priority.
2. **MDD is catastrophic.** -40 to -48% for all attention models. The static ablation gets -31.6%. Attention makes drawdowns WORSE.
3. **Zero regime signal in every new model.** The mission is regime-aware attention. After 9 experiments, no model shows meaningful crisis-vs-calm differentiation in portfolio weights.
4. **Mean pooling in Exp 0007 was a known antipattern.** Philosophy.md explicitly says "no mean/sum/last pooling over time BEFORE the attention layer." Exp 0007 pools AFTER, but the effect is the same — temporal information is destroyed.

---

## Required Changes (before next batch)

1. **Diagnose the EW gap.** Run attention model vs EW per-year comparison. I suspect early years (small training set) are dragging aggregate Sharpe. If so, consider: minimum training window (e.g., start from 2012, not 2008), or weight later years more in aggregation.

2. **Investigate MDD.** Print the portfolio weights and returns during the max drawdown period for Exp 0008. Why -42%? When does it happen? Is the model concentrating into the wrong asset?

3. **Bring back entropy regularization on iTransformer.** Exp 0004 showed entropy reg increases weight variation (even if tautologically). iTransformer already has better base architecture. Combine them. But measure crisis-calm diff honestly — if it doesn't improve, say so.

4. **Try keeping the model from prior year as initialization** (warm-start instead of cold-start). This could help early years and improve regime continuity across windows.

5. **Report EW's MDD.** We need the benchmark drawdown to judge if -42% is model-caused or market-caused.

---

## Questions for the Researcher

1. The iTransformer allocates ~33% SPY, ~20% TLT, ~26% GLD, ~21% SHY — across ALL periods. Why SPY overweight? Is this just learning the equity premium? If so, how is this different from a static tilt?

2. Exp 0007's architecture pools after attention. You know this destroys temporal signal. Why did you try it?

3. The static ablation (Exp 0005) goes 94% SHY in several years. Does the attention model ever produce such extreme allocations? If not, why is the attention model more conservative than 4 learnable parameters?

4. Have you considered that 217-249 params might be in a "dead zone" — too few to learn regime dynamics, but enough to overfit to noise? Would you try either much simpler (< 50 params) or somewhat larger (1000-5000)?

---

**Would I put money in this?** Still no. But the trajectory is improving. The iTransformer architecture is the best foundation so far. The question is no longer "does attention help at all?" (yes, ~0.2 Sharpe) but "can it learn regimes AND beat EW?" That's the right question to be asking.
