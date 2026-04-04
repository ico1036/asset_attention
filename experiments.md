# Experiments

## Exp 0: Linear baseline (DLinear-style)
- Hypothesis: Simple linear is the floor. Window avg → feature pool → linear → softmax.
- Change: Initial baseline
- val_sharpe: 0.84 | test_sharpe: 2.56 | test_mdd: -1.7% | params: 318
- Note: train_sharpe keeps rising while val drops → overfitting even at 318 params. Need regularization or early stopping.
- Verdict: BASELINE (updated with expanding z-score to fix look-ahead bias)
