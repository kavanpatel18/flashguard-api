# Flash Crash Model Accuracy Improvement — Walkthrough

## Problem
The existing GRU model had **0.99 AUC** but was practically useless — **0.01% precision** with **116,936 false alarms** for only 15 true crashes detected.

## Root Causes Identified & Fixed

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | Random `train_test_split` caused **data leakage** across tickers | Time-based 80/20 split per ticker |
| 2 | Extreme class imbalance (94:1) | **SMOTE** oversampling on training set only |
| 3 | Focal loss `alpha=0.25` too low | Dynamic `alpha=0.8994` matching actual ratio |
| 4 | No normalization | **BatchNormalization** after each GRU layer |
| 5 | No attention mechanism | Custom **Bahdanau attention** over timesteps |
| 6 | EarlyStopping on `val_loss` | Now monitors **`val_auc`** with `mode='max'` |
| 7 | Fixed threshold `0.2` | **PR-curve optimal threshold** = `0.617` |

## Results Comparison

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| **Accuracy** | 79.40% | **99.36%** | ▲ +20pp |
| **Precision** | 0.01% | **63.60%** | ▲ 6,360× |
| **Recall** | 100% | **49.60%** | ▼ (by design) |
| **F1-Score** | 0.03% | **55.74%** | ▲ 1,858× |
| **AUC** | 0.990 | **0.891** | ▼ (fair eval, no leakage) |
| **False Alarms** | 116,936 | **107** | ▲ 1,093× fewer |
| **True Positives** | 15 | **187** | ▲ 12.5× more |

> [!IMPORTANT]
> The baseline AUC (0.990) was inflated by data leakage. The new AUC (0.891) is measured on a proper time-based test set with no future data contamination and is a more realistic estimate.

## Files Changed

| File | Action |
|------|--------|
| [train_improved.py](file:///d:/flash%20crash/train_improved.py) | **NEW** — Complete training pipeline |
| [custom_layers.py](file:///d:/flash%20crash/custom_layers.py) | **NEW** — Shared Attention layer class |
| [dashboard.py](file:///d:/flash%20crash/github/dashboard.py) | **MODIFIED** — Added new model to candidates + custom layer loading |
| `improved_flash_crash_model.keras` | **NEW** — Trained model artifact |
| [training_results_v2.json](file:///d:/flash%20crash/training_results_v2.json) | **NEW** — Full metrics report |

## Validation

- ✅ Training completed in 37.7 minutes (50 epochs)
- ✅ Model loads correctly with custom Attention layer: `Input shape: (None, 20, 14)`, `180,033 params`
- ✅ Model copied to `github/` directory for dashboard discovery
- ✅ Dashboard updated with custom object registration

## Live Demo

### Model loaded in dashboard
![Improved model selected and loaded in the dashboard sidebar](C:/Users/kavan/.gemini/antigravity/brain/7e500a61-bc9e-44a0-b4fb-6a380e8f6292/live_prediction.png)

### Portfolio Risk Scanner
![Portfolio scan of 5 NIFTY stocks — all STABLE at ~2.2% risk](C:/Users/kavan/.gemini/antigravity/brain/7e500a61-bc9e-44a0-b4fb-6a380e8f6292/portfolio_scan.png)

### Full Demo Recording
![Dashboard demo walkthrough showing model selection, live prediction, and portfolio scan](C:/Users/kavan/.gemini/antigravity/brain/7e500a61-bc9e-44a0-b4fb-6a380e8f6292/dashboard_model_demo_1773680641417.webp)
