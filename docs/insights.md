
## d_model Scaling Experiment — Key Insight (2026-04-07 23:58 KST)

**Finding**: Larger d_model INCREASES Sharpe but DECREASES regime signal.

| d_model | Mean Sharpe | Mean Shift | Pattern |
|---------|-------------|------------|---------|
| 8 | 0.951 | 6.27% | Best shift, good Sharpe |
| 12 | 0.880 | 8.71% | Highest shift, lower Sharpe |
| 16 | 0.915 | 6.81% | Balanced |
| 24 | 0.931 | 8.51% | Good balance |
| 32 | 0.969 | 5.59% | Best Sharpe, worst shift |

**Paradox**: The models that perform best (high Sharpe) are the least regime-adaptive. They converge to static "smart" allocations (e.g., high SHY weight) that work well on average but don't adapt to market conditions.

**Implication**: The mission's core assumption — that attention can learn implicit regimes — may be fundamentally at odds with Sharpe optimization. The market may not exhibit regime patterns that are:
1. Detectable by attention mechanisms
2. Robust across random initializations
3. Profitable enough to improve Sharpe

---

## Explorer Run: 2026-04-07 00:27 KST — CRON HALTED (STOP + CRITIC HALT)

**Status**: Explorer did NOT run experiments.

**Reason**:
1. Critic review r7_07 directive: "The Explorer MUST NOT run more attention experiments."
2. STOP file exists — User explicitly requested stop on Mon Apr 6 08:31:35 KST 2026

### Current State Summary (unchanged)
- 56+ experiments completed
- ~15 distinct attention architectures tested
- ZERO robust regime detection (all multi-seed validations = 0.01%)
- Mission status: **HALTED — USER STOP + CRITIC HALT**

### Action Taken
- Acquired LOCK at 00:27:57 KST
- Read Critic review r7_07.md — halt order confirmed
- Detected STOP file — user stop request confirmed
- Acknowledged both halt orders — NO experiments run
- Following Exit Protocol without NEEDS_CRITIC (no experiments to review)
- LOCK removed, git commit with status update

### Required: 주인님 Decision
Both Critic and user have halted the mission. Options remain:
1. **Terminate mission** — Document negative result
2. **Pivot to daily rebalancing** — 3× samples
3. **Pivot to explicit regimes** — Use VIX/macros as categorical inputs
4. **Pivot to different architecture** — S4, Mamba, state-space models

**NO further experiments will run until STOP file removed AND human direction provided.**
