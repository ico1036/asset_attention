# Asset Attention

**Attention-based implicit regime learning for ETF asset allocation.**

The mission: build a model where attention weights over time implicitly encode market regimes, producing optimal portfolio weights end-to-end. No categorical regime labels. No human-defined regimes. The attention mechanism *is* the regime detector.

## Architecture

```
17 ETFs × 21yr daily data
    │
    ▼
┌──────────────────────────────────────────┐
│  Attention Model (≤25K params)           │
│  Raw returns → patches → temporal attn   │
│  → cross-asset spatial attn → softmax    │
│  → portfolio weights                     │
│                                          │
│  Regime = implicit in attention weights  │
│  Crisis: attend to vol patches           │
│  Calm: attend to momentum patches        │
└──────────────────────────────────────────┘
    │
    ▼
Walk-forward evaluation (expanding window, 17yr OOS)
```

## Autonomous Research Loop

Two independent AI agents cycle automatically via cron (every 15 min):

```
┌─────────────┐     NEEDS_CRITIC     ┌─────────────┐
│  EXPLORER   │ ──────────────────▶  │   CRITIC    │
│  5 experiments                     │  CIO review  │
│  per session │ ◀──────────────────  │  PASS/REVISE│
└─────────────┘     removes flag     └─────────────┘
        ▲                                    │
        │            CRON (15min)            │
        └────────── orchestrator ────────────┘
```

- **Explorer**: Runs 5 attention experiments, each with one hypothesis. Records results, commits, signals for review.
- **Critic**: Veteran quant CIO persona. Reviews cards + code with fresh eyes. Catches self-deception, overclaiming, mission drift.
- **Cron orchestrator**: Reads state files (`LOCK`, `NEEDS_CRITIC`) and spawns the correct agent. No human in the loop.

### Why 2 agents?
107 experiments across 6 rounds taught us: **a single agent drifts**. The Explorer abandoned attention after MLP "won" in Round 1 and spent 100+ experiments optimizing the wrong thing. A separate Critic with no access to the Explorer's reasoning catches this immediately.

## Project Structure

```
asset_attention/
├── docs/                  ← Agent instructions (DO NOT modify by agents)
│   ├── philosophy.md      ← THE MISSION + constraints
│   ├── program.md         ← 2-agent loop, Explorer/Critic protocols
│   ├── critic.md          ← CIO persona + 8-step review process
│   ├── insights.md        ← Living document: lessons + hypotheses
│   ├── experiments.md     ← Experiment log
│   └── reviews/           ← Critic verdicts (review_rN_MM.md)
├── data/                  ← Price data
│   ├── etf_daily.parquet  ← 17 ETFs × 21yr
│   ├── tensors.pt         ← Preprocessed (built by prepare.py)
│   └── SCHEMA.md
├── cards/                 ← Experiment result cards (JSON)
├── archive/               ← Old experiments (pre-reset, for reference)
├── prepare.py             ← IMMUTABLE evaluation harness
├── guard.py               ← Mission enforcement wrapper
├── train.py               ← ONLY file agents modify
├── LOCK                   ← Present when agent is running
├── NEEDS_CRITIC           ← Present when awaiting Critic review
└── round_tracker.json     ← Current round state
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ≤25K params | ~4,600 samples. Overfit is the enemy. |
| Channel-independent | PatchTST (ICLR 2023): each asset as univariate → multiplies training signal |
| Portfolio-level loss | Academic consensus (Portfolio Transformer, Zhang LSTM) |
| 4-asset pilot first | SPY/TLT/GLD/SHY before scaling to 17 |
| Raw returns only | Attention should learn features. Derived features = human assumptions. |
| Expanding window | Walk-forward: train on past, test on future. 17yr OOS. |
| Sharpe √(50.4) | Weekly rebalancing correction. Previous √252 was 2.24× inflated. |

## Benchmarks (corrected Sharpe, √50.4)

| Strategy | Full | Val | Test |
|----------|------|-----|------|
| Equal Weight | 0.54 | 0.07 | 1.39 |
| LW MinVar w500 | 1.22 | 0.55 | 2.87 |

Any attention model must beat EW (1.39) to justify its complexity.

## Current Status

**Round 7** — First round with mission-enforced framework + 2-agent loop.
- 5 experiments completed (Exp 0000-0004)
- Temperature scaling found critical (entropy 2.43 → 1.14)
- Critic verdict: REVISE — entropy regularization ≠ regime learning
- Next: address Critic feedback, static-weight ablation, multi-seed testing

## Running Locally

```bash
# Setup
uv sync
uv run prepare.py

# Run single experiment (through guard)
uv run guard.py

# The autonomous loop runs via OpenClaw cron — no manual intervention needed
```

## References

- [PatchTST](https://arxiv.org/abs/2211.14730) (ICLR 2023) — Channel-independent patching
- [Portfolio Transformer](https://arxiv.org/abs/2206.03246) (2022) — Direct allocation with GRN
- [iTransformer](https://arxiv.org/abs/2310.06625) (ICLR 2024) — Variables as tokens
- [Crossformer](https://arxiv.org/abs/2209.05249) (ICLR 2023) — Cross-time/cross-variable
- [Signature-Informed Transformer](https://arxiv.org/) (2026) — End-to-end CVaR

## Lessons Learned (Rounds 1-6)

1. **MLP "winning" was fake** — SHY 40% defensive tilt specific to 2020-2026 test period
2. **Seed sensitivity is fatal** — seed=42 gave Sharpe 4.0+, all others converged to EW
3. **Self-review fails** — single agent abandoned mission after 20 experiments
4. **Sharpe bugs compound** — √252 instead of √50.4 hid all problems for 107 experiments
5. **Data scarcity is real** — 839 weekly samples is structurally insufficient for >500 params
