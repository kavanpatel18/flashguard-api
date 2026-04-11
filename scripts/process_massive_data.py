"""
Massive Data Engineering Pipeline
====================================
Step 1 of 2: Data Aggregation

Sources (in order of priority):
  1. `nifty 50 index minute data/` — 22 NIFTY index CSVs, multi-year, 1-min candles
  2. `stk/`                        — 53 individual Nifty 50 stock CSVs
  3. `Demo Crashes/`               — hand-crafted real + synthetic crash scenarios

Strategy:
  - Resample all data to a COMMON RESOLUTION of 5-minute candles
  - Compute STATIONARY features (returns, volatility, z-scores) so data
    from 2015 and 2024 look identical to the model
  - Label crashes using a rolling-window drop threshold
  - Save a single clean, unified CSV ready for model training
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR           = Path(__file__).parent
INDEX_DIR          = BASE_DIR / "nifty 50 index minute data"
STK_DIR            = BASE_DIR / "stk"
DEMO_DIR           = BASE_DIR / "Demo Crashes"
OUTPUT_CSV         = BASE_DIR / "massive_flash_crash_dataset.csv"

RESAMPLE_RULE      = "5min"      # Common time frame for all data sources
SEQUENCE_LENGTH    = 60          # 60 × 5-min = 5 hours of market context per sample

# Crash labelling: flag a candle if close drops ≥ DROP_THRESH within LOOK_FORWARD candles
DROP_THRESH        = 0.005       # 0.5% drop = flash-crash signal
LOOK_FORWARD       = 6           # 6 × 5-min = 30 minutes ahead

# Rolling-window feature lookback (in 5-min candles)
VOL_WINDOW         = 10          # 50-minute rolling volatility
MOM_WINDOW         = 12          # 60-minute momentum window
VWAP_WINDOW        = 30          # 150-minute VWAP context


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_raw_csv(path: Path) -> pd.DataFrame | None:
    """Load a OHLCV CSV file, returning None if unusable."""
    try:
        df = pd.read_csv(path, parse_dates=[0])
        df.columns = [c.lower().strip() for c in df.columns]

        # Normalise the timestamp column name
        ts_col = df.columns[0]
        df = df.rename(columns={ts_col: "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.set_index("datetime").sort_index()

        # Keep only standard OHLCV columns
        keep = {"open", "high", "low", "close", "volume"}
        df = df[[c for c in df.columns if c in keep]]

        if df.empty or len(df) < 100:
            return None

        df = df.astype(float)
        return df
    except Exception as e:
        print(f"  ⚠  Could not load {path.name}: {e}")
        return None


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a DataFrame to 5-minute OHLCV candles."""
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    resampled = df.resample(RESAMPLE_RULE, closed="left", label="left").agg(agg)
    resampled = resampled.dropna(subset=["open", "close"])
    return resampled


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stationary features so era-of-data doesn't matter.
    All features are percentage/deviation based — NOT raw prices.
    """
    c = df["close"]
    h, l, o = df["high"], df["low"], df["open"]

    df["return"]           = c.pct_change()
    df["volatility"]       = df["return"].rolling(VOL_WINDOW, min_periods=2).std()
    df["momentum"]         = c.pct_change(MOM_WINDOW)
    df["high_low_spread"]  = (h - l) / c
    df["open_close_return"]= (c - o) / o

    # Z-score of return (normalises across different eras automatically)
    mu = df["return"].rolling(VOL_WINDOW, min_periods=2).mean()
    df["return_zscore"]    = (df["return"] - mu) / (df["volatility"] + 1e-9)

    # VWAP ratio (close vs rolling VWAP proxy)
    if "volume" in df.columns and df["volume"].sum() > 0:
        typical_price = (h + l + c) / 3
        tp_vol = (typical_price * df["volume"]).rolling(VWAP_WINDOW, min_periods=1).sum()
        vol_sum = df["volume"].rolling(VWAP_WINDOW, min_periods=1).sum()
        vwap = tp_vol / (vol_sum + 1e-9)
        df["vwap_diff"] = (c - vwap) / (vwap + 1e-9)
    else:
        df["vwap_diff"] = 0.0

    df["volume_change"]    = df["volume"].pct_change() if "volume" in df.columns else 0.0

    return df


def label_crashes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-looking binary label:
      1 = the market will drop ≥ DROP_THRESH within LOOK_FORWARD candles
      0 = normal
    """
    future_min = df["close"].shift(-1).rolling(LOOK_FORWARD, min_periods=1).min().shift(-(LOOK_FORWARD - 1))
    pct_drop = (future_min - df["close"]) / df["close"]
    df["crash_label"] = (pct_drop <= -DROP_THRESH).astype(int)
    return df


FEATURE_COLS = [
    "return", "volatility", "momentum", "high_low_spread",
    "open_close_return", "return_zscore", "vwap_diff", "volume_change",
]


def process_file(path: Path, source_tag: str) -> pd.DataFrame | None:
    """Full pipeline for a single CSV: load → resample → features → label."""
    df = load_raw_csv(path)
    if df is None:
        return None

    df = resample_to_5min(df)
    df = engineer_features(df)
    df = label_crashes(df)

    # Drop rows with NaNs in feature columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS)
    df = df.dropna(subset=["crash_label"])

    if len(df) < SEQUENCE_LENGTH + LOOK_FORWARD:
        return None

    df["source"] = source_tag
    df["ticker"] = path.stem

    return df[FEATURE_COLS + ["crash_label", "source", "ticker"]]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    all_frames = []

    # ── 1. NIFTY multi-year index minute data ─────────────────────────────────
    print("\n📊 Source 1: NIFTY Index Minute Data (multi-year)")
    index_files = sorted(INDEX_DIR.glob("*.csv"))
    print(f"   Found {len(index_files)} files")
    for f in tqdm(index_files, desc="  Index CSVs"):
        result = process_file(f, "index")
        if result is not None:
            all_frames.append(result)
            tqdm.write(f"    ✓ {f.name}: {len(result):,} rows, "
                       f"crashes={result['crash_label'].sum()} "
                       f"({100*result['crash_label'].mean():.2f}%)")

    # ── 2. Individual Nifty 50 stock CSVs ─────────────────────────────────────
    print("\n📈 Source 2: Nifty 50 Individual Stocks")
    stk_files = sorted(STK_DIR.glob("*.csv"))
    print(f"   Found {len(stk_files)} files")
    for f in tqdm(stk_files, desc="  Stock CSVs"):
        result = process_file(f, "stock")
        if result is not None:
            all_frames.append(result)
            tqdm.write(f"    ✓ {f.name}: {len(result):,} rows, "
                       f"crashes={result['crash_label'].sum()} "
                       f"({100*result['crash_label'].mean():.2f}%)")

    # ── 3. Demo Crash CSVs (known crash events + synthetics) ──────────────────
    print("\n💥 Source 3: Demo Crashes (Real Events + Systematic Extremes)")
    demo_files = sorted(DEMO_DIR.glob("*.csv"))
    print(f"   Found {len(demo_files)} files")
    for f in tqdm(demo_files, desc="  Demo CSVs"):
        result = process_file(f, "demo_crash")
        if result is not None:
            all_frames.append(result)
            tqdm.write(f"    ✓ {f.name}: {len(result):,} rows, "
                       f"crashes={result['crash_label'].sum()} "
                       f"({100*result['crash_label'].mean():.2f}%)")

    # ── 4. Merge & Save ───────────────────────────────────────────────────────
    if not all_frames:
        print("\n❌ No data could be loaded. Check paths and CSV formats.")
        return

    print("\n🔗 Merging all data sources…")
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=FEATURE_COLS)

    total     = len(combined)
    n_crashes = int(combined["crash_label"].sum())
    n_normal  = total - n_crashes

    print(f"\n{'═' * 65}")
    print(f"  MASSIVE DATASET STATISTICS")
    print(f"{'═' * 65}")
    print(f"  Total rows     : {total:,}")
    print(f"  Crash rows (1) : {n_crashes:,}  ({100*n_crashes/total:.2f}%)")
    print(f"  Normal rows (0): {n_normal:,}  ({100*n_normal/total:.2f}%)")
    print(f"  Unique tickers : {combined['ticker'].nunique()}")
    print(f"  Sources        : {combined['source'].value_counts().to_dict()}")
    print(f"{'═' * 65}")

    combined.to_csv(OUTPUT_CSV, index=False)
    size_mb = OUTPUT_CSV.stat().st_size / 1024 / 1024
    print(f"\n💾 Saved → {OUTPUT_CSV.name}  ({size_mb:.1f} MB)")
    print("\n✅ Data pipeline complete! Now run: train_fast_cnn_transformer.py")


if __name__ == "__main__":
    main()
