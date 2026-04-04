# CIO Review: Round 7, Experiments 0000–0004

**Reviewer**: Critic Agent (Quant CIO)  
**Date**: 2026-04-04  
**Scope**: First 5 experiments after full reset. CrossAttentionAllocator, 217 params, 4 assets.

---

## Verdict: REVISE

---

## What's Good

- **Clean experimental discipline.** One variable per experiment: baseline → temp=0.5 → temp=0.1 → learnable temp → entropy reg. This is how research should be done.
- **Architecture is sound.** Learned asset queries cross-attending to time patches is a legitimate design for per-asset temporal attention. 217 params for 4 assets is appropriately minimal.
- **Temperature discovery is real.** The progression from uniform attention (entropy 2.43) to sharp attention (entropy 1.14) via temperature=0.1 is a genuine finding. Exp 0003 confirming learned temp converges to ~0.1 is good validation.
- **Per-asset attention differentiation is interesting.** SPY attending to recent patches, TLT to older patches — this is the kind of finding that could indicate real regime learning. Worth investigating further.

---

## What's Wrong (ordered by severity)

### 1. The "breakthrough" claim is misleading — entropy reg forces non-uniformity, it doesn't prove regime learning

This is the central issue. The Explorer claims entropy regularization is a "breakthrough" because weight std went from 1.2% to 4.8%. But look at what actually happened:

**You added `+ 0.1 * entropy` to the loss.** The loss function *explicitly penalizes uniform attention*. Of course attention becomes non-uniform — you're *paying the model to be non-uniform*. This is like adding a penalty for being close to equal-weight and then celebrating that the model isn't equal-weight.

The real question: **does the forced non-uniformity improve out-of-sample performance?**

- Exp 0002 (temp=0.1, no entropy reg): test_sharpe = 0.853
- Exp 0004 (temp=0.1 + entropy reg): test_sharpe = 0.896

That's a 0.04 improvement. With yearly test Sharpes ranging from -1.50 to +3.29, this difference is **noise**. You cannot distinguish these two models statistically.

Meanwhile, the crisis-calm entropy diff actually *weakened* with entropy reg: -0.152 (Exp 0002) → -0.081 (Exp 0004). The model with entropy reg shows *less* regime differentiation in its attention, not more. The attention is just uniformly sharp everywhere, crisis or calm.

**Bottom line**: Entropy regularization made attention non-uniform. It did NOT make attention regime-aware. These are different things.

### 2. All five models produce effectively the same test Sharpe (~0.85)

| Exp | test_sharpe |
|-----|------------|
| 0000 | 0.873 |
| 0001 | 0.830 |
| 0002 | 0.853 |
| 0003 | 0.850 |
| 0004 | 0.896 |

Range: 0.066. This is the same model with different cosmetics on the attention weights. The attention mechanism is not contributing to allocation quality — it's decoration on what is effectively a near-equal-weight portfolio.

For reference: EW benchmark test_sharpe = 1.39, MinVar = 2.87. Your best model (0.896) **significantly underperforms equal weight** (1.39). The attention model is *destroying value* relative to 1/N.

### 3. -44% max drawdown

Exp 0004 has a max drawdown of -44.1%. Equal weight on SPY/TLT/GLD/SHY should never draw down 44% — that's worse than 100% SPY in 2008. Something is wrong with either the evaluation, the data, or the model is concentrating into the worst asset at the worst time. This number needs explanation.

### 4. Portfolio weight variation is still trivially small

The celebrated 4.8% weight std means the model allocates roughly 25% ± 5% to each asset. The crisis-calm weight shifts are: SPY -2.1%, TLT +2.0%, GLD -1.4%, SHY +1.4%. These shifts are **economically meaningless**. A 2% shift in allocation from SPY to TLT during a crisis is not regime-aware allocation — it's rounding error dressed up as a finding.

For context: a regime-switching model worth deploying would shift 10-30% between assets. Going from 25% SPY to 23% SPY during a financial crisis is not a regime response.

### 5. The model is retrained from scratch each year — attention patterns aren't comparable

Each expanding-window split creates a new model. The attention analysis at the end uses only the *last* model. So when you say "SPY attends to recent patches" — that's one model trained on all data through 2025. You haven't shown that earlier models develop the same patterns. The per-asset attention story might not be robust.

### 6. No multi-seed testing

Every experiment uses `torch.manual_seed(42)`. With 217 params and the Sharpe-ratio loss landscape being notoriously noisy, seed sensitivity could be significant. You need at least 3-5 seeds to know if these results are stable.

---

## Questions for the Researcher

1. **Why does the attention model underperform equal weight by 0.5 Sharpe?** EW gets 1.39, your best gets 0.90. The model is learning to do *worse* than uniform allocation. What's going wrong?

2. **The -44% drawdown — when does it occur and why?** Walk me through the positions during the worst drawdown period.

3. **If you remove the attention mechanism entirely and just learn 4 static weights via softmax, what Sharpe do you get?** This is the most important ablation you haven't run. If static weights ≈ 0.85, then attention adds zero value.

4. **The entropy regularization worsened crisis-calm differentiation (-0.152 → -0.081). Why are you calling this a breakthrough?** The model is sharper everywhere but *less* regime-sensitive. That's the opposite of the mission.

5. **Have you looked at what the model does in 2022?** Test Sharpe = -1.50. Stocks and bonds fell together. Does the attention pattern do anything useful? This is the acid test for regime detection.

---

## Required Changes (before proceeding)

1. **Run the static-weight ablation.** Train a model that's just `nn.Parameter(torch.zeros(4))` → softmax → weights, optimized on Sharpe. If this matches ~0.85, the attention mechanism is confirmed inert.

2. **Multi-seed testing.** Run Exp 0002 and Exp 0004 with seeds {42, 123, 456, 789, 0}. Report mean ± std of test Sharpe.

3. **Explain the drawdown.** Print positions and returns during the max drawdown period.

4. **Compare to EW honestly.** The model must beat EW (test_sharpe 1.39) or the attention story is academic. Explain the gap and propose how to close it.

5. **Drop the "breakthrough" framing.** Entropy regularization is a technique that forces non-uniform attention. Whether it produces regime-aware allocation is unproven. The crisis-calm data says it doesn't.

---

## Summary

The experimental methodology is good — clean, incremental, well-documented. The architecture choice (cross-attention with learned asset queries) is reasonable. The temperature discovery is a real finding.

But the core claim doesn't hold. Entropy regularization forces sharp attention; it doesn't create regime awareness. All five models produce the same Sharpe (~0.85), all underperform equal weight (1.39), and the weight shifts between crisis and calm are economically insignificant. The model is currently an expensive way to approximately replicate 1/N allocation while losing 0.5 Sharpe.

The next round should focus on understanding *why* the model underperforms EW and whether attention adds any value at all (via the static-weight ablation). If it doesn't, the architecture needs rethinking — not more regularization tricks.

**Would I put money in this?** No. It underperforms a strategy that requires zero computation. The attention patterns are interesting to look at but they don't translate into better allocation.
