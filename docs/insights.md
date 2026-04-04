# Insights

_Fresh start after major refactor. Round 7+ focuses exclusively on attention-based regime learning._

## Mission
Build an attention model that implicitly learns market regimes and outputs optimal asset weights.

## Benchmarks (from prepare.py, corrected Sharpe with sqrt(50.4))
- EW: full=0.54, val=0.07, test=1.39
- MinVar w500: full=1.22, val=0.55, test=2.87

## Lessons from Rounds 1-6 (107 experiments)

### What failed and why
1. **MLP with mean(dim=time)**: Collapsed time axis → learned static SHY/UUP bias, not regimes. Seed-dependent.
2. **Spatial Attention alone**: No temporal information → can't learn regimes.
3. **PatchTST / Temporal Attention**: Right idea (preserve time), but 839 samples caused overfitting. Val↑ test↓.
4. **Dual Attention**: Combined spatial+temporal, but too many params for data size.
5. **All Sharpe numbers from R1-6 were ~2.24x inflated** (sqrt(252) bug, fixed in refactor).

### Key insight from R1-6
With 839 samples, any model >500 params overfits. Daily rebalancing gives ~4600 samples.

## Round 7, Batch 1 (Exp 0000-0004) — Critic Verdict: REVISE

### What we learned
- **Temperature 0.1 sharpens attention** (entropy 2.43 → 1.14). Real finding.
- **Per-asset attention patterns differ**: SPY→recent, TLT→distant. Interesting but unvalidated.

### ⚠️ Critic Corrections (MUST address before next batch)
1. **Entropy reg ≠ regime learning.** Penalizing uniform attention forces non-uniformity — that's tautological. Crisis-calm diff actually WORSENED (-0.152 → -0.081). Drop the "breakthrough" framing.
2. **All 5 models ≈ test_sharpe 0.85, but EW = 1.39.** The attention model DESTROYS value vs equal weight. This is the #1 problem.
3. **±2% weight shifts are economically meaningless.** Real regime models shift 10-30%.
4. **MDD -44% unexplained.** Worse than 100% SPY. Something is wrong.
5. **No multi-seed testing.** Seed=42 only. Previous rounds showed seed sensitivity is fatal.

### Required experiments (next batch):
1. **Static-weight ablation**: `nn.Parameter(torch.zeros(4))` → softmax → weights, trained on Sharpe. If test_sharpe ≈ 0.85, attention adds zero value.
2. **Multi-seed test**: Run best model with seeds {42, 123, 456, 789, 0}. Report mean ± std.
3. **MDD investigation**: Print positions during max drawdown period.
4. **Honest EW comparison**: Why does the model underperform EW by 0.5 Sharpe? Diagnose.

## Open Questions
- Is 217 params simply too few for meaningful attention?
- Does cross-attention architecture create a bottleneck? (learned queries may be too constrained)
- Would temporal self-attention (PatchTST-style) work better than cross-attention?
- Is the expanding window approach causing the model to be trained on too little data in early years?

## Round 7, Batch 2 (Exp 0005-0008) — Critic Verdict: REVISE

### What we learned
- **Static ablation: attention adds ~0.2 Sharpe** (0.652 → 0.87). Small but real.
- **Multi-seed: CrossAttention is robust.** 0.869 ± 0.030 across 5 seeds. Not a seed artifact.
- **PatchSelfAttention (mean pool) = dead.** Mean pooling after attention destroys temporal signal. Zero regime signal.
- **iTransformer is best architecture so far.** test_sharpe 0.904 with asset embeddings + temporal self-attention. But regime signal still near-zero — it found better STATIC weights (overweight SPY), not regime dynamics.
- **All attention models: MDD -40 to -48%.** Static ablation: -31.6%. Attention makes drawdowns WORSE.

### ⚠️ Critic Corrections (MUST address before next batch)
1. **Diagnose EW gap (0.9 vs 1.39).** Run per-year comparison. Hypothesis: early years with small training sets drag aggregate. Consider minimum training window or warm-start.
2. **Investigate MDD.** Print weights and returns during max drawdown for Exp 0008. Why -42%?
3. **Combine entropy reg + iTransformer.** Best architecture + best regularization. Measure crisis-calm diff honestly.
4. **Try warm-start** (initialize from prior year's model instead of cold-start each year).
5. **Report EW's MDD** as benchmark.

### Key insight
The EW underperformance may be a **training protocol problem**, not an attention problem. Expanding window with tiny initial training sets produces bad models for early years, dragging aggregate Sharpe. This must be investigated.

## Next Hypotheses (after addressing Critic)
1. **Diagnose per-year Sharpe vs EW** — find where the gap comes from
2. **iTransformer + entropy reg** — combine best architecture with best regularization trick
3. **Warm-start training** — carry model weights across expanding windows
4. **Minimum training window** — don't start until enough data exists (e.g., 5+ years)
