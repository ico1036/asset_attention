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

### What was NEVER properly tried
- ~~Attention with sufficient data (daily rebal = ~3100 samples, or augmentation)~~  ← now using daily rebal
- ~~Attention with extreme simplicity (d_model=4-8, 1 head, under 500 params)~~ ← now 217 params
- ~~Regime signal analysis~~ ← now analyzing entropy per year
- Loss functions designed for regime sensitivity → Exp 0004 added entropy reg

### Key insight from R1-6
The previous rounds proved that **with 839 samples, any model >500 params overfits**. Daily rebalancing gives ~4600 samples.

## Round 7 Discoveries (Exp 0000-0004)

### Temperature is critical for attention sharpness
- Default temperature (1.0) → nearly uniform attention (entropy 2.43/2.49 max)
- temp=0.5 → modest improvement (entropy 2.28)
- **temp=0.1 → sharp, asset-specific attention (entropy 1.14)**
- Learnable temperature converges to ~0.1, confirming this is the right range

### Each asset develops distinct temporal focus (temp=0.1)
- **SPY** → attends to patches 8-9 (most recent weeks)
- **TLT** → attends to patches 0-1 and 6-7 (oldest and mid-period)
- **SHY** → attends to patches 0-1 (oldest)
- **GLD** → more distributed attention
- This is intuitive: equities respond to recent momentum, bonds to longer-term trends

### Entropy regularization is the key to portfolio variation
- Sharp attention alone (temp=0.1) → weight std ~1.2% (barely different from EW)
- Adding entropy penalty to loss → weight std ~4.8% (4x improvement)
- The bottleneck was score_proj + portfolio softmax smoothing out the attention signal

### All models produce test_sharpe ~0.85
- This is consistent across all 5 experiments regardless of attention sharpness
- Suggests the cross-attention architecture with 217 params has a Sharpe ceiling
- Need either more params, richer features, or architectural changes to improve Sharpe

## Next Hypotheses
1. **Larger d_model (16-32)** with temp=0.1 + entropy reg — does capacity help?
2. **Multi-head attention** (2-4 heads) — different heads could capture different regime signals
3. **Remove score_proj bottleneck** — use attention output directly as logits (reduce indirection)
4. **Contrastive regime loss** — explicitly encourage different attention patterns for different volatility regimes
5. **Increase entropy reg coefficient** — push weight variation further
