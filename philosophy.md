# Philosophy

## Why Attention for Asset Allocation?
- Attention weights = implicit regime encoding. No categorical labels (bull/bear/sideways).
- End-to-end: data → portfolio weights directly. No predict-then-optimize pipeline.
- Softmax output = fully invested long-only allocation.

## Design Principles
1. **Data-honest**: ~5,000 independent time points. Model must be ≤25K params (ratio ≥2:1).
2. **Simplicity**: Karpathy MicroGPT philosophy. Strip everything unnecessary.
3. **Price-based features only**: No lagging macro (CPI/GDP = dead values when ffilled to daily). Use VIX, credit spread, yield curve, momentum, vol — all derivable from price.
4. **Dual Attention**: Spatial (feature↔feature cross-correlation) + Temporal (time patterns). Single attention only captures one axis.
5. **Patching**: Compress days into patches (e.g., 5-day). Reduces sequence length, preserves local patterns.
6. **Modern building blocks**: RoPE, SwiGLU, RMSNorm — free upgrades from LLM research.

## Overfitting is the Enemy (Lesson from Round 1)
- val_sharpe↑ while test_sharpe↓ = overfitting. The simplest MLP (692 params) had best test_sharpe (3.94).
- Primary metric: **test_sharpe** (not val_sharpe). Val is for early stopping only.
- Secondary: **val-test gap** — minimize it. Gap > 1.0 = suspicious.
- Goal: beat Equal Weight benchmark on TEST set (EW test_sharpe ≈ 2.76).

## Complexity Must Be Earned
- Baseline: Linear (DLinear). If attention can't beat linear, it's not needed.
- Add complexity one step at a time: Linear → MLP → Single Attention → Dual Attention
- Each step must show val_sharpe improvement. No improvement = revert to simpler.
- Ref: "Are Transformers Effective for Time Series Forecasting?" (Zeng 2023, AAAI)

## What to Explore
- Attention order: spatial→temporal vs temporal→spatial vs interleaved
- Patch size: 3, 5, 10 days
- Loss: -Sharpe, -Sharpe + turnover penalty, CVaR
- Architecture alternatives: MLP-Mixer, pure temporal-only, pure spatial-only

## Data Integrity
- **Look-ahead bias**: Only use information available at decision time.
  - Market-traded prices (VIX, yields, ETF prices): same-day OK (known at close).
  - Monthly macro (CPI, GDP, unemployment): +30 day publication lag minimum.
  - Weekly macro (jobless claims): +7 day lag.
  - Never use future returns in feature computation.
- **Survivorship bias**: All 17 ETFs still trade today. If adding new assets, verify listing date.
- **Z-score normalization**: Must be expanding window (train only), never full-sample.
- **Walk-forward**: Train on past, validate on unseen future. No shuffling. Time order sacred.
  - Phase 1 (current): Single sequential split (70/15/15). Simple and fast.
  - Phase 2 (later): Rolling window retrain for robustness check.
- **Rebalancing cost**: Assume 5bps per turnover as baseline. Turnover penalty in loss reflects this.
- **Suspiciously good results**: If val_sharpe > 2.0, assume bug until proven otherwise. Check:
  1. Is future data leaking into features? (z-score, returns)
  2. Is the model just memorizing a few samples?
  3. Is turnover unrealistically high? (free lunch = no free lunch)
  4. Does it degrade on a different time split?
  5. Compare with equal weight — if model Sharpe >> EW Sharpe, something is wrong.
  6. Check IS (train) vs OOS (val/test) gap — if train_sharpe >> val_sharpe, it's overfitting, not alpha.

## What NOT to Do
- No models >25K params
- No external macro data as input features
- No two-stage predict→optimize
- No categorical regime labels
- No overfitting excuses — walk-forward is the judge

## Key References
- iTransformer (ICLR 2024): variables-as-tokens = our spatial attention
- PatchTST (ICLR 2023): patching time series
- Crossformer (ICLR 2023): two-stage cross-time/cross-variable
- Signature-Informed Transformer (2026): end-to-end CVaR, path signatures
- Portfolio Transformer (2022): direct allocation, ETF 7개 with similar data size

## Data
- 17 ETFs × ~21yr daily (SPY, QQQ, IWM, EFA, VEA, VWO, EEM, TLT, IEF, SHY, TIP, HYG, GLD, DBC, USO, VNQ, UUP)
- 10 features: VIX, VIX Δ5d, SPY-TLT corr 20d, 10Y-2Y spread, 10Y Δ20d, HYG-TLT spread, SPY/200MA, GLD/SPY Δ20d, momentum, realized vol
- Platform: Mac Mini M4, 32GB, MPS backend
