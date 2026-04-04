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
2. Form a hypothesis. Write it as a comment at the top of `train.py`.
3. Modify `train.py`. You may change anything: model architecture, optimizer, loss, hyperparameters.
4. Run: `uv run train.py` — fixed 5-minute wall clock budget. This auto-saves a card to `cards/exp_NNNN.json`.
   Add model-specific details to the config dict in train.py (e.g., n_layers, d_model, n_heads, attention_order).
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

## Self-Diagnosis (run before every experiment)
- If last experiment finished in < 60s: data is too small or model too simple. Fix data pipeline first.
- If val-test gap > 1.5: overfitting. Add regularization or reduce model.
- If val_sharpe > 2.0: suspect bug. Run sanity checks per philosophy.md.
- If all seeds except one give poor results: seed-dependent, not real alpha.
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

## DreamWalk (Harness Integrity Check)

When `philosophy.md`, `program.md`, or `prepare.py` changes:
1. Mentally simulate 5 experiments + 1 Dream step-by-step.
2. At each step, ask: "Can I do this? Is something missing? Will it break?"
3. **Quantitative sanity check**:
   - Estimate training time: n_samples × n_params × epochs → will it fill the time budget?
   - If estimated time < 30s: data pipeline or model size needs fixing BEFORE experiments.
   - Check data/param ratio: samples/params < 5 = overfitting guaranteed.
   - Check if val/test split sizes are statistically meaningful (< 50 samples = unreliable).
4. Fix any issues found before running real experiments.

## Dream Phase (AutoDream)

Every 10 experiments, pause and "dream":
1. Read all of `experiments.md` and `cards/*.json`.
2. Extract patterns: what consistently helps, what consistently fails, what's untested.
3. Write/update `insights.md` with distilled learnings.
4. Before each new experiment, check `insights.md` — don't repeat known failures.

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
