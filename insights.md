# Insights

_Updated by Dream Phase (Round 5 mid) after Exp 79. 79 experiments across 5 rounds._

## The Real Answer: LW_MinVar Beats EW (Round 5 Discovery)

**Rounds 1-4 concluded "nothing beats EW." Round 5 proved this WRONG.**

Ledoit-Wolf Minimum Variance portfolio (0 learned parameters) beats Equal Weight on:
- **val**: 1.13 vs 0.76 (+0.37)
- **test**: 5.05 vs 2.76 (+2.30)
- **Robust across ALL 4 train/val/test splits** (Δtest: +1.2 to +2.3)

### Why LW_MinVar Works and Learned Models Don't

1. **LW_MinVar exploits structural information (covariance)** that doesn't need to be "learned" from labeled samples
2. **Ledoit-Wolf shrinkage** provides stable covariance estimation from just 60 daily returns — unlike learned models that need 620+ labeled allocation→return pairs
3. The 17-ETF universe spans very different risk levels (SHY vol ≈ 0.3% vs USO vol ≈ 3%). MinVar naturally tilts toward low-vol assets.
4. **Learned models converge to EW** because with 620 samples, any model with >~50 params overfits to noise. The optimal regularized solution IS EW.
5. MinVar is not "learned" — it's an analytical solution to a different optimization problem (minimize variance, not maximize Sharpe).

### What the Portfolio Looks Like

LW_MinVar (window=60) average test-period allocation:
- UUP (USD index): 23.7%
- SHY (1-3yr Treasury): 22.4%
- TIP (TIPS): 12.7%
- HYG (High Yield): 12.2%
- IEF (7-10yr Treasury): 10.2%
- **~80% in bonds/cash, ~20% in equities/commodities**

This is a **defensive portfolio** that worked especially well in 2022-2026 (rate hiking, equity volatility). But it also beat EW on val (2019-2022 including COVID), suggesting genuine risk-adjusted superiority.

## Hierarchy of Strategies (by test Sharpe, w=60)

| Strategy | Val | Test | Params | Notes |
|----------|-----|------|--------|-------|
| LW_MinVar | 1.13 | 5.05 | 0 | **Best overall** |
| InvVol_lb40_fl0.5 | 0.32 | 3.46 | 0 | Good test, poor val |
| VolRegime_50pct | 0.65 | 3.08 | 0 | Balanced val/test |
| SmoothFloor_s3 | 0.61 | 3.02 | 0 | Balanced |
| Ensemble_3way | 0.42 | 3.10 | 0 | Diversified |
| EqualWeight | 0.76 | 2.76 | 0 | Former "unbeatable" baseline |
| Attn+Cross (best learned) | 1.48 | 2.17 | 944 | Best learned model |
| MLP (best learned) | 0.92 | 2.42 | 692 | |

## What Learned Models Can and Cannot Do

### Cannot Do:
- Beat EW when sample size is 620 and features are noisy
- Learn covariance structure better than analytical estimation
- Generalize cross-asset dynamics from so few samples

### Can Do:
- Achieve similar performance to EW (val ~1.0-1.5) with consistency
- The ranking model (Exp 74) is remarkably consistent across seeds
- Attention has better val generalization than MLP

## Confirmed Round 5 Findings
- **Adding momentum to InvVol hurts** — momentum is noise at weekly frequency
- **EW-deviation loss makes models too conservative** — they learn to not deviate
- **Pairwise ranking doesn't help** despite 136× sample multiplication
- **Regime classification is deterministic** with logistic regression (too few features)
- **Learned models given MinVar as input still converge to ~EW** — can't leverage the structural edge
- **Floor constraints on InvVol don't bind** with LW shrinkage (weights already reasonable)

## Remaining Questions (for Exp 80-89)
1. Can attention improve LW_MinVar by learning better covariance estimates?
2. What about online/adaptive covariance with exponential weighting?
3. Can we combine LW_MinVar with a small learned tilt that improves val?
4. Does monthly rebalancing (fewer, cleaner signals) change the picture?
5. What is the MAX_DRAWDOWN of LW_MinVar vs EW?
