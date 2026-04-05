# CIO Review: Round 7 Batch 5 (Exp 0031-0034)

## Verdict: **PARTIAL SUCCESS** — Continue Mission

---

### Summary
After 35 experiments with zero regime detection, **Exp 0031 and 0032 finally produced meaningful regime signals** — 11-13% crisis-calm weight shifts. This is the first evidence that attention-based implicit regime learning is possible with this data and architecture.

The breakthrough came not from new architecture but from **hyperparameter tuning** — specifically, d_model values of 4 and 16 produce regime signals while the baseline d=8 produces near-zero.

---

### What's Good

**1. Regime Signal Detected (FINALLY)**
- **Exp 0031 (d=4)**: max_shift = **11.1%** — Crisis overweights GLD (+11%) and SPY (+3.5%), underweights TLT (-7.9%) and SHY (-6.8%)
- **Exp 0032 (d=16)**: max_shift = **13.1%** — Crisis overweights GLD/SHY (risk-off), underweights SPY/TLT
- This is economically meaningful differentiation, not statistical noise

**2. Crisis/Calm Allocations Make Sense**
- d=4 model: Crisis = [33% SPY, 14% TLT, 36% GLD, 17% SHY] — Growth assets overweight with bond underweight (crisis alpha-seeking?)
- d=16 model: Crisis = [19% SPY, 16% TLT, 36% GLD, 29% SHY] — Classic risk-off with gold/cash overweight
- Both are defensible strategies, just different risk postures

**3. Sharpe in Acceptable Range**
- 0.83-0.93 test_sharpe across all 4 experiments — consistent with previous batches
- Still below EW (1.39), but regime detection is the primary mission metric now

**4. Hyperparameter Discovery**
- The non-monotonic relationship (d=4 works, d=8 doesn't, d=16 works) is a real finding
- Suggests the loss landscape has "islands" of good regime detection

---

### What's Wrong

**1. Sharpe Still Below EW by ~0.5**
- EW = 1.39, best model = 0.93 — attention still destroys 33% of risk-adjusted return
- Regime detection exists but isn't translating to better performance
- Possible explanations: (a) regime signal too weak, (b) wrong regime detection, (c) transaction costs, (d) overfitting to some periods

**2. MDD Still -34% to -43%**
- d=16 model shows -34.5% MDD (improved from -42% baseline)
- But still worse than what a simple bond-heavy static allocation would achieve

**3. Non-Monotonic d_model Effect is Unexplained**
- Why does d=4 work but d=8 doesn't? This needs an explanation
- Hypothesis: d=4 is underfitting to noise, d=8 is overfitting, d=16 has enough capacity to learn real signal
- But this is speculation — needs validation

**4. Temperature and Entropy Effects Are Weak**
- temp=0.05: 8.1% shift (down from 0.1's ~11%)
- entropy=0.5: 4.2% shift (down from 0.1's ~11%)
- Baseline hyperparameters are already near-optimal

**5. No Multi-Seed Validation**
- All results from seed=42 only
- Given history of seed sensitivity, this is a major risk
- These "regime signals" could be seed artifacts

---

### Required Changes for Next Batch

**CRITICAL — Do Not Skip:**

1. **Multi-seed validation of 0031 and 0032**
   - Run d=4 and d=16 models with seeds {42, 123, 456, 789, 0}
   - Report: (a) mean/max_shift, (b) std of max_shift, (c) whether regime signal persists across seeds
   - If regime signal disappears with different seeds → false positive, architecture still broken

2. **Systematic d_model search**
   - Test d_model ∈ {2, 4, 6, 8, 12, 16, 24, 32}
   - Plot: d_model vs max_shift, d_model vs test_sharpe
   - Identify if there's a "sweet spot" region

3. **Sharpe vs Regime Tradeoff Analysis**
   - Exp 0032 has better regime (13.1%) but worse Sharpe (0.83)
   - Exp 0031 has moderate regime (11.1%) and better Sharpe (0.93)
   - Is there a tradeoff? Or can we get both?

4. **Regime Signal Stability Over Time**
   - Current max_shift is computed from aggregate crisis vs calm periods
   - Does the regime signal appear consistently across all crisis years (2008, 2009, 2020, 2022)?
   - Or is it driven by one specific crisis?

---

### Questions for the Researcher

1. **Why does d=8 fail while d=4 and d=16 succeed?** What's the mechanism?
2. **Are the crisis/calm weight shifts economically sensible?** d=4 overweights SPY in crisis — is this alpha-seeking or a bug?
3. **What happens with multi-seed testing?** If these results are seed-dependent, they're not real.
4. **Can we get regime shift >15% without sacrificing Sharpe?** 11-13% is meaningful but still modest.

---

### Strategic Assessment

| Hypothesis | Prior | Updated | Evidence |
|------------|-------|---------|----------|
| H1: Training protocol | 15% | 10% | Not the main issue |
| H2: Attention wrong approach | 85% | 55% | **Regime signal detected!** |
| H3: Hyperparameter sensitivity | — | **35%** | Non-monotonic d_model effect is real |

**The mission is no longer at critical risk.** Attention-based regime learning is possible — we have proof of concept. The question is now: can we make it robust and economically viable?

**Next decision point:** After multi-seed validation and d_model search, if regime signal holds:
- **Yes**: Continue with optimization (loss function, ensemble, daily rebalancing)
- **No**: Return to FAIL status and consider mission pivot

---

### Final Verdict

**PARTIAL SUCCESS** — First regime signal detected after 35 experiments. Continue mission with systematic hyperparameter validation.

*Do not declare victory. Do not get distracted by new architectures. Validate these results rigorously.*
