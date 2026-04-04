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

## Next Hypotheses (after addressing Critic)
1. **Static ablation first** — establish if attention adds ANY value
2. **Daily rebal exploration** — ~4600 samples vs current setup
3. **PatchTST self-attention** instead of cross-attention — richer temporal modeling
4. **GRN gating** (from Portfolio Transformer) — adaptive complexity for small data
