# Experiments

## Exp 0: Linear baseline (DLinear-style)
- Hypothesis: Simple linear is the floor. Window avg → feature pool → linear → softmax.
- Change: Initial baseline
- val_sharpe: 0.84 | test_sharpe: 2.56 | test_mdd: -1.7% | params: 318
- Note: train_sharpe keeps rising while val drops → overfitting even at 318 params. Need regularization or early stopping.
- Verdict: BASELINE (updated with expanding z-score to fix look-ahead bias)

## Exp 1: MLP with hidden layer + dropout + early stopping
- Hypothesis: Single hidden layer MLP captures non-linear feature interactions the linear baseline misses.
- Change: MLPAllocator(hidden=32, dropout=0.3), cosine LR, grad clip, early stopping patience=50
- val_sharpe: 1.18 | test_sharpe: 3.94 | test_mdd: -0.8% | params: 692
- Verdict: KEEP
