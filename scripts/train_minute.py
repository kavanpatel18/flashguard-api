"""
Minute-Level Flash Crash GRU Training Script
==============================================
Trains a BiGRU+Attention model on NIFTY index minute-bar data.

Adaptations from the daily model:
  • No volume/VWAP/turnover features (index data has volume=0)
  • Flash crash = ≥2% drop over a 30-min window + high volatility
  • 10 price-only features engineered from OHLC
  • Uses all 22 NIFTY index CSVs for diverse training data
"""

import json, time, os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Bidirectional, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.regularizers import l2

# import shared Attention layer
import sys
_parent = str(Path(__file__).resolve().parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from custom_layers import Attention

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠  imblearn not installed — SMOTE disabled.")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR          = Path(__file__).with_name("nifty 50 index minute data")
OUTPUT_MODEL_PATH = Path(__file__).with_name("improved_minute_model.keras")
OUTPUT_META_PATH  = Path(__file__).with_name("training_results_minute.json")
BEST_CKPT_PATH    = Path(__file__).with_name("best_minute_model.keras")

SEQUENCE_LENGTH = 30       # 30-minute lookback window
CRASH_WINDOW    = 30       # look 30 minutes ahead for price drop
CRASH_THRESHOLD = -0.02    # ≥2% drop = flash crash
VOL_QUANTILE    = 0.90     # volatility must be in top 10%

EPOCHS     = 40
BATCH_SIZE = 512
TRAIN_RATIO = 0.80

# 10 price-only features (no volume/VWAP/turnover)
FEATURES = [
    "return", "log_return",
    "volatility_5", "volatility_10", "volatility_20",
    "momentum_5", "momentum_10",
    "high_low_spread", "open_close_return",
    "price_acceleration",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce    = -(y_true * tf.math.log(y_pred) +
                   (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t    = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss_fn


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (price-only, no volume)
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build price-only features from OHLC minute bars."""
    out = df.copy()

    # basic returns
    out["return"]       = out["close"].pct_change()
    out["log_return"]   = np.log(out["close"] / out["close"].shift(1))

    # multi-scale volatility
    out["volatility_5"]  = out["return"].rolling(5).std()
    out["volatility_10"] = out["return"].rolling(10).std()
    out["volatility_20"] = out["return"].rolling(20).std()

    # momentum
    out["momentum_5"]  = out["close"].pct_change(5)
    out["momentum_10"] = out["close"].pct_change(10)

    # price structure
    out["high_low_spread"]   = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["open_close_return"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    # acceleration (change of momentum)
    out["price_acceleration"] = out["return"].diff()

    return out


def label_crashes(df: pd.DataFrame) -> pd.DataFrame:
    """Label flash crashes: ≥2% drop over 30min + high volatility."""
    out = df.copy()

    # forward-looking price drop over CRASH_WINDOW minutes
    future_close = out["close"].shift(-CRASH_WINDOW)
    price_drop = (future_close - out["close"]) / out["close"]

    # crash = big drop + high volatility
    vol_threshold = out["volatility_10"].quantile(VOL_QUANTILE)
    out["crash_label"] = (
        (price_drop <= CRASH_THRESHOLD) &
        (out["volatility_10"] > vol_threshold)
    ).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load all NIFTY minute CSVs, engineer features, label crashes."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    print(f"📂 Found {len(csv_files)} CSV files in {DATA_DIR}")

    all_dfs = []
    for fp in csv_files:
        idx_name = fp.stem.replace("_minute", "")
        print(f"   Loading {idx_name}…", end=" ")
        df = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["index_name"] = idx_name

        df = engineer_features(df)
        df = label_crashes(df)
        df = df.dropna(subset=FEATURES).reset_index(drop=True)

        n_crash = df["crash_label"].sum()
        print(f"rows={len(df):,}  crashes={n_crash}")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 Combined: {len(combined):,} rows")
    print(f"   Crash distribution:\n{combined['crash_label'].value_counts().to_string()}")
    return combined


def build_sequences(combined: pd.DataFrame):
    """
    Memory-efficient sequence builder.
    Uses strided subsampling: keeps ALL crash sequences but
    only every STRIDE-th non-crash sequence to fit in memory.
    """
    from sklearn.preprocessing import StandardScaler

    STRIDE = 10  # skip 9 out of 10 non-crash sequences

    X_train_all, y_train_all = [], []
    X_test_all,  y_test_all  = [], []

    for idx_name in combined["index_name"].unique():
        sdf = combined[combined["index_name"] == idx_name].reset_index(drop=True)

        # scale features per index
        scaler = StandardScaler()
        vals = scaler.fit_transform(sdf[FEATURES].values).astype(np.float32)
        labels = sdf["crash_label"].values

        n_seq = len(vals) - SEQUENCE_LENGTH
        if n_seq <= 0:
            continue

        # Identify which sequence indices are crash vs non-crash
        seq_labels = np.array([labels[i + SEQUENCE_LENGTH - 1] for i in range(n_seq)])

        crash_idxs     = np.where(seq_labels == 1)[0]
        non_crash_idxs = np.where(seq_labels == 0)[0]

        # subsample non-crash: every STRIDE-th
        non_crash_sub = non_crash_idxs[::STRIDE]

        # combine and sort
        keep_idxs = np.sort(np.concatenate([crash_idxs, non_crash_sub]))

        # build sequences only for kept indices
        seqs = np.array([vals[i:i + SEQUENCE_LENGTH] for i in keep_idxs],
                        dtype=np.float32)
        labs = seq_labels[keep_idxs]

        split_idx = int(len(seqs) * TRAIN_RATIO)
        X_train_all.append(seqs[:split_idx])
        y_train_all.append(labs[:split_idx])
        X_test_all.append(seqs[split_idx:])
        y_test_all.append(labs[split_idx:])

        print(f"   {idx_name}: kept {len(keep_idxs):,} / {n_seq:,} seqs "
              f"(crash={len(crash_idxs):,}, sampled_normal={len(non_crash_sub):,})")

    X_train = np.concatenate(X_train_all)
    y_train = np.concatenate(y_train_all)
    X_test  = np.concatenate(X_test_all)
    y_test  = np.concatenate(y_test_all)

    print(f"\n   Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"   Train labels → {Counter(y_train)}")
    print(f"   Test  labels → {Counter(y_test)}")
    return X_train, y_train, X_test, y_test


def apply_smote(X_train, y_train):
    if not HAS_SMOTE:
        print("⚠  Skipping SMOTE.")
        return X_train, y_train

    n, t, f = X_train.shape
    X_flat = X_train.reshape(n, t * f)

    ratio = Counter(y_train)
    if ratio[1] < 2:
        print("⚠  Too few positive samples for SMOTE, skipping.")
        return X_train, y_train

    target_count = min(ratio[0] // 5, ratio[1] * 10)
    target_count = max(target_count, ratio[1] * 3)

    k = min(5, ratio[1] - 1)
    print(f"🔄 SMOTE  (minority {ratio[1]:,} → {target_count:,}, k={k}) …")
    sm = SMOTE(sampling_strategy={1: target_count}, random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_flat, y_train)
    X_res = X_res.reshape(-1, t, f).astype(np.float32)
    print(f"   After SMOTE: {Counter(y_res)}")
    return X_res, y_res


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(timesteps: int, n_features: int) -> Model:
    inp = Input(shape=(timesteps, n_features), name="input")

    x = Bidirectional(GRU(96, return_sequences=True, recurrent_dropout=0.1),
                      name="bi_gru_1")(inp)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.3, name="drop_1")(x)

    x = GRU(48, return_sequences=True, recurrent_dropout=0.1, name="gru_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.3, name="drop_2")(x)

    x = Attention(name="attention")(x)

    x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4), name="dense_1")(x)
    x = Dropout(0.2, name="drop_3")(x)
    out = Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inp, outputs=out, name="minute_bigru_attention")


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD  &  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(y_true, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    best = np.argmax(f1s)
    t = float(thresholds[best])
    print(f"\n🎯 Optimal threshold: {t:.4f}  "
          f"F1={f1s[best]:.4f}  P={prec[best]:.4f}  R={rec[best]:.4f}")
    return t


def evaluate(y_true, y_prob, threshold, label=""):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc  = (tp + tn) / len(y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)

    print(f"\n{'═'*60}\n  {label}  (threshold={threshold:.4f})\n{'═'*60}")
    print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"  AUC={auc:.4f}  AP={ap:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "accuracy": float(acc), "precision": float(prec),
        "recall": float(rec), "f1": float(f1),
        "auc": float(auc), "average_precision": float(ap),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MINUTE-LEVEL FLASH CRASH GRU TRAINING")
    print("=" * 70)

    # 1. Load & engineer
    combined = load_all_data()

    # 2. Build sequences
    X_train, y_train, X_test, y_test = build_sequences(combined)

    # 3. SMOTE
    X_train, y_train = apply_smote(X_train, y_train)

    # 4. Focal loss alpha
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    alpha = neg / (pos + neg)
    print(f"\n📊 Focal loss alpha: {alpha:.4f}  (pos={pos:,}, neg={neg:,})")

    # 5. Build & compile
    model = build_model(SEQUENCE_LENGTH, len(FEATURES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=alpha),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # 6. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(BEST_CKPT_PATH), monitor="val_auc",
                        mode="max", save_best_only=True, verbose=1),
    ]

    # 7. Train
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks, verbose=1,
    )
    train_time = time.time() - t0
    print(f"\n⏱  Training: {train_time:.1f}s  ({train_time/60:.1f} min)")

    # 8. Predict
    y_prob = model.predict(X_test, batch_size=1024, verbose=0).ravel()
    y_prob = np.clip(y_prob, 0.0, 1.0)

    # 9. Threshold
    optimal_thresh = find_optimal_threshold(y_test, y_prob)

    # 10. Evaluate
    metrics = evaluate(y_test, y_prob, optimal_thresh, "OPTIMAL THRESHOLD")
    for t in [0.5, 0.3, 0.2, 0.1]:
        evaluate(y_test, y_prob, t, f"FIXED THRESHOLD {t}")

    # 11. Save
    model.save(str(OUTPUT_MODEL_PATH))
    print(f"\n💾 Model saved → {OUTPUT_MODEL_PATH}")

    result = {
        "model_type": "Minute_BiGRU_Attention",
        "data_level": "minute",
        "optimal_threshold": optimal_thresh,
        "training_time_seconds": round(train_time, 2),
        "training_time_minutes": round(train_time / 60, 2),
        "total_parameters": int(model.count_params()),
        "sequence_length": SEQUENCE_LENGTH,
        "num_features": len(FEATURES),
        "crash_window_minutes": CRASH_WINDOW,
        "crash_threshold_pct": CRASH_THRESHOLD,
        "smote_applied": HAS_SMOTE,
        "focal_loss_alpha": round(alpha, 4),
        "epochs_run": len(history.history["loss"]),
        "num_index_series": 22,
        **{f"test_{k}": v for k, v in metrics.items()},
    }
    OUTPUT_META_PATH.write_text(json.dumps(result, indent=2))
    print(f"📋 Results → {OUTPUT_META_PATH}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
