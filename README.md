# Asset Attention: MicroAllocator

Autonomous research system for ETF asset allocation using AI agents.  
Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — adapted for portfolio optimization.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        HUMAN (주인님)                        │
│  • Updates philosophy.md / program.md                       │
│  • Reviews results via main session                         │
└──────────────┬──────────────────────────┬───────────────────┘
               │ Out-of-Loop              │ direction
               │ DreamWalk                │ changes
               ▼                          │
┌──────────────────────────┐              │
│      MAIN SESSION        │◄─────────────┘
│  • Validates harness      │
│  • Spawns experiment agent│
│  • Reports results        │
└──────────┬───────────────┘
           │ sessions_spawn
           ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXPERIMENT AGENT (isolated)                │
│                                                             │
│  ┌─────────┐    ┌──────────────────────────────────────┐   │
│  │ LOCK    │    │         EXPERIMENT LOOP               │   │
│  │ create  │───▶│                                      │   │
│  └─────────┘    │  1. Read insights.md (learned lessons)│   │
│                 │  2. Self-Diagnosis (check last result) │   │
│                 │  3. Hypothesis + Expected Range        │   │
│                 │  4. Modify train.py                    │   │
│                 │  5. uv run guard.py ◄── HARD CHECKS   │   │
│                 │     │                                  │   │
│                 │     ▼                                  │   │
│                 │  ┌──────────┐  ┌───────────────────┐  │   │
│                 │  │ train.py │──▶ cards/exp_NNNN.json│  │   │
│                 │  └──────────┘  └───────┬───────────┘  │   │
│                 │                        │              │   │
│                 │     ┌──────────────────▼──────────┐   │   │
│                 │     │        guard.py             │   │   │
│                 │     │  • Params ≤ 25K?            │   │   │
│                 │     │  • Train time reasonable?    │   │   │
│                 │     │  • Loss curve healthy?       │   │   │
│                 │     │  • val_sharpe in sane range? │   │   │
│                 │     │  • IS/OOS gap < 1.5?        │   │   │
│                 │     │  • Auto verdict: KEEP/DISCARD│   │   │
│                 │     └──────────────────────────────┘   │   │
│                 │                                        │   │
│                 │  6. KEEP → commit & push               │   │
│                 │     DISCARD → revert train.py          │   │
│                 │  7. Record in experiments.md            │   │
│                 │                                        │   │
│                 │  ┌─────────────────────────────┐       │   │
│                 │  │ Every 10 exp: IN-LOOP DREAM │       │   │
│                 │  │  • actual vs expected ranges │       │   │
│                 │  │  • pattern extraction        │       │   │
│                 │  │  • update insights.md        │       │   │
│                 │  └─────────────────────────────┘       │   │
│                 │                                        │   │
│                 │  Repeat until 20 exp or early stop     │   │
│                 └────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────┐                                                │
│  │ LOCK    │                                                │
│  │ remove  │                                                │
│  └─────────┘                                                │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Owner | Purpose |
|------|-------|---------|
| `philosophy.md` | Human | Design principles, constraints, what NOT to do |
| `program.md` | Human | Experiment loop, rules, DreamWalk protocol |
| `prepare.py` | Fixed | Parquet → raw tensors (one-time) |
| `train.py` | Agent | Model + features + training (freely modified) |
| `guard.py` | Fixed | Hard checks that cannot be bypassed |
| `experiments.md` | Agent | Experiment log with verdicts |
| `insights.md` | Agent | Distilled learnings from Dream phases |
| `cards/exp_NNNN.json` | Auto | Full config + results + verdict per experiment |
| `data/SCHEMA.md` | Fixed | Data schema with column names and lag info |

## DreamWalk Protocol

Two types of integrity checks:

- **Out-of-Loop**: When harness files change → main session simulates 5 experiments → fixes issues before agent runs
- **In-Loop**: Every 10 experiments → agent compares actual vs expected → stops if systemic issues found

## Guard Checks (Automated)

1. **Params**: ≤ 25K
2. **Training time**: Must match expected time for model complexity × data size
3. **Loss curves**: train_loss must fall; val_loss must not diverge
4. **Sanity bounds**: val_sharpe ∈ [-0.5, 4.0], test_sharpe ∈ [-0.5, 5.0]
5. **IS/OOS gap**: |val - test| < 1.5
6. **Suspicious results**: val_sharpe > 2.0 triggers investigation

## Results (55 experiments, 3 rounds)

**Best model**: MLP + GELU + Noise (692 params, val=1.85, test=4.23)  
**Benchmark**: Equal Weight Sharpe = 2.76

Key finding: Attention mechanisms do not improve over simple MLP for 17-ETF allocation with ~620 samples. The MLP's apparent alpha is a period-specific defensive tilt (SHY-heavy), not learned cross-asset dynamics.

See `insights.md` for full analysis.

## Setup

```bash
uv sync
uv run prepare.py    # one-time data prep
uv run guard.py      # run experiment through guard checks
```

## License

MIT
