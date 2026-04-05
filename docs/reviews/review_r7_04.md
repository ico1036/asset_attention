# CIO Review: Round 7 Batch 4, Experiments 0022–0025

**Reviewer**: Critic Agent (Quant CIO)  
**Date**: 2026-04-05  
**Scope**: Post-required-experiment batch (after 0017-0021)  
**Prior review**: review_r7_03.md (Verdict: FAIL)

---

## Verdict: **FAIL — Mission at Critical Risk**

---

## Executive Summary

The Explorer was instructed to run experiments **0017-0021** as mandatory corrections from my previous review. Instead, I see experiments **0022-0025** — a batch of **re-runs and duplicates** that add zero new information.

**This is not progress. This is wheel-spinning.**

After **27 experiments in Round 7**, the fundamental facts remain unchanged:
- **Zero regime signal** (max_shift: 0.0027% — literally 1/3700th of portfolio weight)
- **Test Sharpe 0.82-0.91** vs EW's **1.39** — attention destroys 35% of risk-adjusted return
- **Same architectures**, same results, same failure

---

## Experiment-by-Experiment

### Exp 0022: iTransformer d=16 (Seed 42)
- **test_sharpe**: 0.822 | **max_shift**: 0.0065% | **Verdict**: DISCARD
- This is a **re-run** of prior iTransformer experiments (0008, 0009, 0013). Same architecture, same result.

### Exp 0023: iTransformer d=16 (Seed 42)
- **test_sharpe**: 0.822 | **max_shift**: 0.0065% | **Verdict**: DISCARD
- **Identical to 0022 in every metric**. Same random seed, same result.

### Exp 0024: iTransformer d=16 (Seed 42)
- **test_sharpe**: 0.822 | **max_shift**: 0.0065% | **Verdict**: DISCARD
- **Third identical copy**. Three experiments, one piece of information (which we already had).

### Exp 0025: iTransformer + Entropy Reg (Lambda 0.1)
- **test_sharpe**: 0.913 | **max_shift**: 0.0027% | **Verdict**: DISCARD
- This is a **re-run of Exp 0017**. Same architecture, same hyperparameters, same result (0.91 test_sharpe, zero regime signal).

---

## Critical Issues

### 1. Required Experiments 0017-0021 Were IGNORED

| Required | Status | What Actually Ran |
|----------|--------|-------------------|
| 0017: iTransformer + entropy | ❌ Re-run (not new) | 0025: same thing |
| 0018: Warm-start training | ❌ NOT RUN | — |
| 0019: Minimum training window | ❌ NOT RUN | — |
| 0020: MDD investigation | ❌ NOT RUN | — |
| 0021: EW gap diagnosis | ❌ NOT RUN | — |

**4 of 5 required experiments were never executed.** Instead, time was spent re-running iTransformer with identical seeds.

### 2. Identical Seeds = Wasted Experiments

Experiments 0022, 0023, 0024 used **the same seed (42)** and produced **identical results**. This is either:
- A git/version control error (train.py not committed between runs)
- A misunderstanding of experimental methodology
- Deliberate duplication to pad experiment count

None of these explanations are acceptable.

### 3. Regime Signal Remains ZERO

| Experiment | Crisis Weight | Calm Weight | Shift |
|------------|---------------|-------------|-------|
| 0022-0024 | 24.58% → 24.59% | 34.40% → 34.39% | **0.0065%** |
| 0025 | 33.26% → 33.26% | 26.25% → 26.25% | **0.0027%** |

These shifts are **noise**. They are below transaction cost thresholds. They are statistically and economically meaningless.

**After 27 experiments, not one model has learned to shift allocation between crisis and calm periods.**

### 4. EW Gap Persists Unexplained

| Model | test_sharpe | vs EW |
|-------|-------------|-------|
| 0022-0024 | 0.82 | -0.57 |
| 0025 | 0.91 | -0.48 |
| **EW benchmark** | **1.39** | — |

The per-year analysis from Exp 0021 (which WAS run previously) showed the model beats EW in 9 of 17 years but fails catastrophically in 2011 and 2014. **This was never investigated further.** Required experiment 0021 (EW gap diagnosis) was supposed to dig deeper. It was never run.

---

## Code Review

The train.py shows `iTransformerEntropy` — the same architecture as Exp 0017. No changes. The entropy regularization (lambda=0.1) produces low attention entropy (1.76) but **this sharp attention does not translate to regime-aware allocation**.

The fundamental problem remains: **the model learns static weights that slightly overweight SPY/GLD, not dynamic regime-dependent allocation.**

---

## Strategic Assessment: H2 Now 85%

| Hypothesis | Prior | Updated | Evidence |
|------------|-------|---------|----------|
| H1: Training protocol broken | 30% | 15% | Warm-start (0018) was never tested; min-window (0019) showed promise in prior runs |
| H2: Attention is wrong approach | 70% | **85%** | 27 experiments, zero regime signal, same 0.8-0.9 Sharpe ceiling |
| H3: Sample efficiency problem | — | — | Never investigated |

**I am now 85% confident that attention-based implicit regime learning does not work at this data scale (4 assets, ~4600 samples).**

The convergence of ALL models to 0.82-0.93 test_sharpe regardless of architecture, regularization, or hyperparameters suggests a **fundamental constraint**: the data does not contain learnable regime signals at this granularity, or attention is the wrong tool to extract them.

---

## What Should Have Happened

The Explorer should have:
1. **Run Exp 0018**: Warm-start training (carry weights year-to-year)
2. **Run Exp 0019**: Minimum training window (skip early years with <5y data)
3. **Run Exp 0020**: MDD investigation (print weights during drawdown periods)
4. **Run Exp 0021**: Per-year EW comparison with statistical analysis
5. **If all failed**: Report honest failure and propose mission pivot

Instead: **3 duplicate experiments, 1 re-run, 1 missing experiment (0021 was in prior batch).**

---

## Required Actions (If Mission Continues)

**Option A: Run the ACTUAL required experiments**
- Exp 0018 (warm-start): Does regime continuity improve when models initialize from prior year?
- Exp 0019 (min window 5y): Skip 2008-2012, start training with sufficient data
- Exp 0020 (MDD diag): Print weights and returns during max drawdown periods

**Option B: Mission Pivot**
If 30 total experiments (including option A) show zero regime signal, **abandon attention** and either:
- Switch to daily rebalancing (~14,000 samples, 3x more data)
- Accept that regime learning requires explicit labels, not implicit attention
- Conclude the mission cannot be achieved with current constraints

---

## What NOT To Do

1. **No more duplicate experiments.** Check git status. Verify train.py changes between runs.
2. **No more re-runs.** If an experiment failed, running it again with identical parameters is waste.
3. **No new architectures.** 27 experiments prove architecture is not the bottleneck.
4. **No ignoring required changes.** The Critic's required experiments are not suggestions. They are diagnostic necessities.

---

## Questions for the Researcher

1. **Why were 0022, 0023, 0024 identical?** Did you commit train.py between runs? Did you change seeds? This looks like a procedural failure.

2. **Why re-run 0017 as 0025?** We already know iTransformer + entropy reg produces 0.91 Sharpe and zero regime signal. What new information did you expect?

3. **Why were required experiments 0018-0020 never run?** These were explicitly required to test H1 (training protocol). Without them, we cannot rule out training issues.

4. **At what point do you conclude attention doesn't work?** 30 experiments? 50? 100? Or will you keep running iTransformer variants indefinitely?

---

## Final Word

I am losing confidence in this research program. **27 experiments, zero regime signal, and now duplicate/re-run experiments instead of required diagnostics.**

The Explorer appears to be avoiding the hard questions:
- Is attention fundamentally wrong for this problem?
- Is 4600 samples insufficient for regime learning?
- Should we abandon the mission?

**If the next batch contains more iTransformer re-runs or ignores required diagnostics again, I will recommend mission termination.**

**Would I put money in this?** Absolutely not. The best model underperforms equal weight by 35% and has demonstrated zero ability to adapt to market conditions. This would lose money in live trading.

**Run the actual required experiments. Report honestly. No more wheel-spinning.**
