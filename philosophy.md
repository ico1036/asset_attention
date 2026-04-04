# Philosophy

## THE MISSION (inviolable)

**Build an attention-based model that implicitly learns market regimes through its attention weights and outputs optimal asset allocation weights end-to-end.**

This is the entire reason this project exists. Everything below serves this mission.

### What this means concretely:
1. The model MUST contain attention mechanism(s).
2. Attention weights MUST operate over a time dimension (not collapsed by mean/sum).
3. The model takes raw time-series as input and outputs portfolio weights via softmax.
4. Regime detection is implicit — encoded in how attention weights shift over time. No categorical regime labels.
5. The model must learn WHEN to allocate WHERE — temporal dynamics, not static feature averages.

### What is NOT acceptable:
- Models without attention (MLP, Linear, MinVar, etc.) are BENCHMARKS, not solutions.
- A benchmark beating the attention model does NOT mean "give up on attention." It means the attention model needs more work.
- Collapsing the time axis (mean, sum, last) before the attention layer defeats the purpose. The attention must SEE the time series.
- Declaring the project "solved" by a non-attention method.

## Design Principles (under the mission)

1. **Simplicity serves the mission**: Strip unnecessary complexity FROM THE ATTENTION MODEL. Don't replace attention with something simpler. Make the attention architecture itself as clean as possible.
2. **Data-honest**: ≤25K params. If data is insufficient for the current architecture, fix the data problem (more assets, higher frequency, augmentation) — don't abandon the architecture.
3. **Price-based features only**: No lagging macro indicators.
4. **Dual Attention**: Spatial (cross-asset) + Temporal (time patterns). Test both orders and interleaved.
5. **Patching**: Compress days into patches to give temporal attention meaningful chunks.

## Data Scarcity is a Problem to SOLVE, Not a Reason to Quit

If 839 samples aren't enough for attention to learn:
- Increase rebalancing frequency (daily = ~3100 samples)
- Expand asset universe (more cross-sectional variation)
- Data augmentation (noise injection, bootstrap, time-shift)
- Reduce model size further
- Try different input representations

"Not enough data" is NEVER a valid final conclusion. It's a problem statement.

## Evaluation

- Primary metric: val_sharpe (for model selection), test_sharpe (for final judgment)
- Benchmarks (for comparison only, NOT targets to beat): Equal Weight, MinVar, MLP
- A model with lower Sharpe than MinVar but valid attention-based regime detection is MORE valuable than MinVar — because it can improve with more data. MinVar cannot.
- Secondary: visualize attention weights across different market periods. Do they change? Do they make sense?

## Regime Detection Quality (must report)

Every attention experiment MUST include in its card:
- Attention weight visualization or summary statistics across 3+ distinct market periods
- Weight entropy: does it change over time? (static attention = not learning regimes)
- Portfolio composition shift: does allocation change meaningfully between calm and volatile periods?

## What NOT to Do
- No models >25K params
- No external macro data as input features
- No two-stage predict→optimize
- No categorical regime labels
- No declaring victory with non-attention models

## Key References
- iTransformer (ICLR 2024): variables-as-tokens = spatial attention
- PatchTST (ICLR 2023): patching time series for temporal attention
- Crossformer (ICLR 2023): two-stage cross-time/cross-variable
- Signature-Informed Transformer (2026): end-to-end CVaR, path signatures
- Portfolio Transformer (2022): direct allocation, 7 ETFs

## Data
- 17 ETFs × ~21yr daily
- Platform: Mac Mini M4, 32GB, MPS backend
