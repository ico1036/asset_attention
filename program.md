# Program

Read `philosophy.md` first. It defines WHY we're doing this and the constraints.

## Lock Protocol
Before starting experiments, create `LOCK` file. Remove it when done.
If `LOCK` exists, check its timestamp. If older than 1 hour, it's stale — delete and proceed. Otherwise wait or abort.
```bash
if [ -f LOCK ]; then echo "LOCKED"; exit 1; fi
echo "$(date)" > LOCK
# ... run experiments ...
rm LOCK
```

## Setup (once per session)
```bash
uv sync
uv run prepare.py   # builds tensors from parquet, ~10s
```

## Experiment Loop

1. Read the current `train.py` and `experiments.md` (past results).
2. Form a hypothesis AND expected range. Write both at the top of `train.py`:
   ```
   # Hypothesis: Adding attention should capture cross-asset dynamics
   # Expected: val_sharpe 1.0-1.5 (vs baseline 0.84), train_time 60-120s
   # If val_sharpe > 3.0 or < 0.0 or train_time < 30s → investigate before continuing
   ```
3. Modify `train.py`. You may change anything: model architecture, optimizer, loss, hyperparameters.
4. Run: `uv run guard.py` (NOT train.py directly) — this runs train.py + enforces hard checks. Never bypass guard.py.
   Add to the config dict in train.py:
   - Model details (n_layers, d_model, n_heads, attention_order)
   - `loss_curve: {"train": [...], "val": [...]}` — sample at [0%, 25%, 50%, 75%, 100%] of training
   - `expected: {"val_sharpe": [lo, hi], "train_time": [lo, hi]}` — your prediction before running
5. Record result in `experiments.md` (use same exp number as the card):
   ```
   ## Exp N: [short description]
   - Hypothesis: ...
   - Change: ...
   - val_sharpe: X.XX | val_mdd: X.XX% | params: XXK
   - Verdict: KEEP / DISCARD
   ```
6. If KEEP: `git add -A && git commit -m "exp N: [description]" && git push`.
   If DISCARD: `git checkout HEAD -- train.py` to restore last KEEP version. Card stays (failures are data too).
7. If train.py crashes, record as FAILED in experiments.md, revert train.py, and continue.
8. Repeat from step 1.

## Session Limits
- Default: run up to 20 experiments per session, then stop and remove LOCK.
- Or stop when time budget (e.g., 3 hours) is reached.
- Always remove LOCK on exit, even on error (use try/finally).

## Self-Respawn (mandatory on session end)
When your session limit (20 exp or 3hr) is reached, your LAST actions before exiting must be:
1. Remove LOCK file.
2. Update `insights.md` with this round's key findings.
3. Append a round summary to `experiments.md`.
4. `git add -A && git commit -m "round N complete" && git push`
5. Send a respawn request to the main session:
   ```
   sessions_send(sessionKey="main", message="Autoresearch round complete. Please spawn next round. Last exp: N, insights updated.")
   ```
   If `sessions_send` is unavailable, write a file `RESPAWN_REQUEST` with the same info — the cron watchdog will pick it up.
This ensures continuity. A cron watchdog also runs every 45 minutes as a safety net for crash recovery.

## Self-Diagnosis (run before every experiment)
- Read `insights.md` first — it overrides generic rules with learned lessons.
- If last experiment finished too fast (see guard.py estimates): investigate why. Check insights.md for known causes before changing data pipeline.
- If val-test gap > 1.5: overfitting. Add regularization or reduce model.
- If val_sharpe > 2.0: suspect bug. Run sanity checks per philosophy.md.
- If all seeds except one give poor results: seed-dependent, not real alpha.
- NEVER stop early because "nothing works." If current approach hits a wall, pivot: change the problem formulation, data representation, loss function, or training method. Think creatively. Use all 20 experiments.
- Do NOT ask the human. Diagnose and fix autonomously.

## Rules
- Only modify `train.py`. Never touch `prepare.py`.
- `train.py` may read any file in `data/` (parquets, tensors, metadata). This is not "modifying prepare.py".
- Max 25K parameters. Print param count at start.
- Metric: `val_sharpe` (higher = better). Secondary: `val_mdd` (lower = better).
- Device: MPS (Apple Silicon). Use `torch.device("mps")`.
- Walk-forward validation: train on past, test on unseen future. No peeking.
- Every experiment must be reproducible (set seed).
- If you need a new library, run `uv add <package>` first. Update pyproject.toml before importing.

## DreamWalk (Two Types)

### Out-of-Loop DreamWalk (harness changes)
Triggered when: `philosophy.md`, `program.md`, `guard.py`, or `prepare.py` changes.
Owner: the human or main session (NOT the experiment agent).
Purpose: verify the entire workflow still works before sending the agent back in.
1. Mentally simulate 5 experiments + 1 Dream step-by-step.
2. At each step, ask: "Can I do this? Is something missing? Will it break?"
3. **Quantitative sanity check**:
   - Estimate training time: n_samples × n_params × epochs → will it fill the time budget?
   - If estimated time < 30s: data pipeline or model size needs fixing BEFORE experiments.
   - Check data/param ratio: samples/params < 5 = overfitting guaranteed.
   - Check if val/test split sizes are statistically meaningful (< 50 samples = unreliable).
4. **Pre-experiment prediction**: For each simulated experiment, predict expected val_sharpe range, training time, and likely failure modes.
5. **Post-experiment comparison**: After simulated results, check: "Is this within my prediction? If not, why?"
6. Fix any issues found before running real experiments.

### In-Loop DreamWalk (during experiments)
Triggered when: Dream Phase runs (every 10 experiments).
Owner: the experiment agent.
Purpose: verify experiments are producing sensible results.
1. Compare all recent cards' actual results vs their `expected` ranges.
2. If >50% of experiments fell outside expected ranges → something systemic is wrong. Stop and investigate.
3. Check: are loss curves healthy? Is the complexity ladder making sense?
4. Update `insights.md` with pattern corrections.

## Dream Phase (AutoDream)

Every 10 experiments, pause and "dream":
1. Read all of `experiments.md` and `cards/*.json`.
2. Extract patterns: what consistently helps, what consistently fails, what's untested.
3. **Ask "WHY?"**: Why did failures fail? Why did successes succeed? What's the root cause?
4. **Ask "WHAT IF?"**: Can the problem be reformulated? Different loss? Different target? Different data representation? Think like a researcher, not a hyperparameter tuner.
5. **Generate 3 new hypotheses** that are fundamentally different from what's been tried. Write them in `insights.md` under "Next Hypotheses."
6. Write/update `insights.md` with patterns + new hypotheses.
7. Before each new experiment, check `insights.md` — don't repeat known failures, DO try the new hypotheses.

## What to Try
- Attention order variations (spatial→temporal, temporal→spatial, interleaved)
- Patch sizes (3, 5, 10 days)
- Loss functions (-Sharpe, -Sharpe + turnover, CVaR)
- Depth (1-3 layers)
- Head count (1, 2, 4)
- d_model (16, 32, 64)
- Dropout, weight decay
- Alternative architectures (MLP-Mixer, linear, CNN)
- Activation functions (SwiGLU, GELU, ReLU)
- Position encoding (RoPE, learned, sinusoidal, none)
