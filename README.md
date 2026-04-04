# Asset Attention: MicroAllocator

Dual-Attention Transformer for ETF asset allocation.

## Architecture

**MicroAllocator v2** — ~15-25K parameters, designed for data-scarce regime-aware allocation.

```
Input: 17 ETFs × 60 days × 10 price-based features
  ↓
[Patch Embedding] 60 days → 12 patches (5-day each)
  ↓
[Spatial Attention] Feature-to-feature cross-correlation (per patch)
  ↓
[Temporal Attention] Patch-to-patch time patterns (per asset)
  ↓
[Portfolio Head] Softmax → asset weights
  ↓
Loss: -Sharpe + turnover penalty + CVaR
```

### Key Design Choices

- **End-to-end**: No predict-then-optimize pipeline. Direct data → weights.
- **Dual Attention**: Spatial (feature cross-correlation) + Temporal (time patterns)
- **Patching** (PatchTST): 5-day patches reduce sequence length 5×, preserve local patterns
- **RoPE**: Rotary position embedding for relative temporal encoding
- **SwiGLU + RMSNorm**: Modern LLM building blocks, free performance gain
- **Price-based features only**: VIX, credit spread, yield curve, momentum, vol — no lagging macro

### Inspired By

| Paper | Technique Borrowed |
|-------|--------------------|
| iTransformer (ICLR 2024) | Variables-as-tokens spatial attention |
| PatchTST (ICLR 2023) | Time series patching |
| Crossformer (ICLR 2023) | Two-stage cross-time/cross-variable attention |
| Signature-Informed Transformer (2026) | End-to-end CVaR, path signature features |
| Portfolio Transformer (2022) | Direct allocation via attention |
| DeepSeek V3 | RoPE, SwiGLU, RMSNorm |

### Data

- **17 ETFs** × ~21 years daily (SPY, QQQ, IWM, EFA, VEA, VWO, EEM, TLT, IEF, SHY, TIP, HYG, GLD, DBC, USO, VNQ, UUP)
- **10 price-based features**: VIX level, VIX Δ5d, SPY-TLT corr 20d, 10Y-2Y spread, 10Y rate Δ20d, HYG-TLT spread, SPY/200MA, GLD/SPY Δ20d, asset momentum, realized vol 20d
- Walk-forward validation

### Constraints

- **Mac Mini M4** (32GB unified, 10 GPU cores, Metal/MPS)
- **~15-25K params** (data ratio ~2-3:1 with regularization)
- Training target: < 30 min

## Setup

```bash
uv sync
```

## Project Structure

```
asset_attention/
├── data/
│   ├── collect_data.py       # Data collection pipeline
│   ├── etf_daily.parquet     # 17 ETFs × 21yr daily
│   ├── macro_daily.parquet   # 12 macro indicators (reference)
│   └── etf_metadata.json     # Asset class/region metadata
├── pyproject.toml
└── README.md
```

## License

MIT
