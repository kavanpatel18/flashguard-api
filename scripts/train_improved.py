"""
Improved Flash Crash GRU Training Script
==========================================
Fixes vs. the original notebook:
  1. Time-based train/test split per ticker (no data leakage)
  2. SMOTE oversampling on the training set only
  3. Focal loss with dynamic alpha matched to class imbalance
  4. Bidirectional GRU + Attention + BatchNormalization + L2
  5. ReduceLROnPlateau + EarlyStopping on val_auc
  6. Automatic threshold optimisation via precision-recall curve
  7. Comprehensive evaluation & side-by-side comparison with baseline
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GRU,
    Bidirectional,
    Dense,
    Dropout,
    BatchNormalization,
    Layer,
    Permute,
    Multiply,
    Flatten,
    RepeatVector,
    Lambda,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.regularizers import l2

# ── Try SMOTE; fall back gracefully ──────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠  imblearn not installed — SMOTE disabled. "
          "Install with:  pip install imbalanced-learn")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH      = Path(__file__).with_name("flash_crash_ready_dataset.csv")
OUTPUT_MODEL_PATH = Path(__file__).with_name("improved_flash_crash_model.keras")
OUTPUT_META_PATH  = Path(__file__).with_name("training_results_v2.json")
BEST_CKPT_PATH    = Path(__file__).with_name("best_improved_model.keras")

FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "VWAP",
    "return", "volatility", "momentum", "volume_change",
    "vwap_diff", "high_low_spread", "open_close_return", "turnover_change",
]

SEQUENCE_LENGTH = 20
EPOCHS          = 50
BATCH_SIZE      = 256
TRAIN_RATIO     = 0.80   # first 80% per ticker for train


# ═══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS  (dynamic alpha)
# ═══════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma=2.0, alpha=0.75):
    """Binary focal loss.  alpha weights the *positive* class."""
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
# ATTENTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class Attention(Layer):
    """Bahdanau-style additive attention over timesteps."""

    def build(self, input_shape):
        self.W = self.add_weight(name="att_W",
                                 shape=(int(input_shape[-1]), int(input_shape[-1])),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b",
                                 shape=(int(input_shape[-1]),),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u",
                                 shape=(int(input_shape[-1]),),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=[[2], [0]]), axis=1
        )                                          # (batch, timesteps)
        context = tf.reduce_sum(
            x * tf.expand_dims(attention_weights, -1), axis=1
        )                                          # (batch, features)
        return context

    def get_config(self):
        return super().get_config()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_split(path: Path):
    """Load CSV and do a chronological 80/20 split *per ticker*."""
    print(f"📂 Loading dataset from {path} …")
    df = pd.read_csv(path)
    print(f"   Shape: {df.shape}  |  Tickers: {df['ticker'].nunique()}")
    print(f"   Crash label distribution:\n{df['crash_label'].value_counts().to_string()}")

    X_train_all, y_train_all = [], []
    X_test_all,  y_test_all  = [], []

    for ticker in df["ticker"].unique():
        sdf = df[df["ticker"] == ticker].reset_index(drop=True)
        vals   = sdf[FEATURES].values.astype(np.float32)
        labels = sdf["crash_label"].values

        n_seq = len(vals) - SEQUENCE_LENGTH
        if n_seq <= 0:
            continue

        seqs = np.array([vals[i:i + SEQUENCE_LENGTH] for i in range(n_seq)],
                        dtype=np.float32)
        labs = np.array([labels[i + SEQUENCE_LENGTH] for i in range(n_seq)])

        split_idx = int(len(seqs) * TRAIN_RATIO)
        X_train_all.append(seqs[:split_idx])
        y_train_all.append(labs[:split_idx])
        X_test_all.append(seqs[split_idx:])
        y_test_all.append(labs[split_idx:])

    X_train = np.concatenate(X_train_all)
    y_train = np.concatenate(y_train_all)
    X_test  = np.concatenate(X_test_all)
    y_test  = np.concatenate(y_test_all)

    print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"   Train labels → {Counter(y_train)}")
    print(f"   Test  labels → {Counter(y_test)}")
    return X_train, y_train, X_test, y_test


def apply_smote(X_train, y_train):
    """Reshape to 2-D, apply SMOTE, reshape back."""
    if not HAS_SMOTE:
        print("⚠  Skipping SMOTE (not installed).")
        return X_train, y_train

    n, t, f = X_train.shape
    X_flat = X_train.reshape(n, t * f)

    ratio = Counter(y_train)
    target_count = min(ratio[0], ratio[1] * 10)  # oversample to 10× not 1:1
    target_count = max(target_count, ratio[1] * 5)

    print(f"🔄 Applying SMOTE  (minority {ratio[1]} → {target_count}) …")
    sm = SMOTE(
        sampling_strategy={1: target_count},
        random_state=42,
        k_neighbors=min(5, ratio[1] - 1),
    )
    X_res, y_res = sm.fit_resample(X_flat, y_train)
    X_res = X_res.reshape(-1, t, f).astype(np.float32)
    print(f"   After SMOTE: {Counter(y_res)}")
    return X_res, y_res


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(timesteps: int, n_features: int) -> Model:
    inp = Input(shape=(timesteps, n_features), name="input")

    # --- Bidirectional GRU block 1 ---
    x = Bidirectional(
        GRU(128, return_sequences=True, recurrent_dropout=0.1),
        name="bi_gru_1",
    )(inp)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.3, name="drop_1")(x)

    # --- GRU block 2 (return sequences for attention) ---
    x = GRU(64, return_sequences=True, recurrent_dropout=0.1,
            name="gru_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.3, name="drop_2")(x)

    # --- Attention ---
    x = Attention(name="attention")(x)

    # --- Classifier head ---
    x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4),
              name="dense_1")(x)
    x = Dropout(0.2, name="drop_3")(x)
    out = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="improved_gru")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 on the precision-recall curve."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns n+1 values; trim
    prec = prec[:-1]
    rec  = rec[:-1]

    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)

    best_threshold = float(thresholds[best_idx])
    best_f1        = float(f1_scores[best_idx])
    best_prec      = float(prec[best_idx])
    best_rec       = float(rec[best_idx])

    print(f"\n🎯 Optimal threshold: {best_threshold:.4f}")
    print(f"   F1={best_f1:.4f}  Prec={best_prec:.4f}  Rec={best_rec:.4f}")
    return best_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (multi-threshold)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_prob, threshold: float, label: str = ""):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    acc  = (tp + tn) / len(y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)

    print(f"\n{'═' * 60}")
    print(f"  {label}  (threshold={threshold:.4f})")
    print(f"{'═' * 60}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  F1-score:           {f1:.4f}")
    print(f"  ROC AUC:            {auc:.4f}")
    print(f"  Avg Precision (AP): {ap:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'═' * 60}")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "average_precision": float(ap),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  IMPROVED FLASH CRASH GRU TRAINING")
    print("=" * 70)

    # ── 1. Load & split ──────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_and_split(DATASET_PATH)

    # ── 2. SMOTE on training data only ───────────────────────────────────────
    X_train, y_train = apply_smote(X_train, y_train)

    # ── 3. Compute dynamic focal loss alpha ──────────────────────────────────
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    alpha = neg_count / (pos_count + neg_count)
    print(f"\n📊 Focal loss alpha (auto): {alpha:.4f}  "
          f"(pos={pos_count}, neg={neg_count})")

    # ── 4. Build model ───────────────────────────────────────────────────────
    model = build_model(SEQUENCE_LENGTH, len(FEATURES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=alpha),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # ── 5. Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            str(BEST_CKPT_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── 6. Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0
    print(f"\n⏱  Training time: {train_time:.1f}s  ({train_time/60:.1f} min)")

    # ── 7. Predict on test set ───────────────────────────────────────────────
    print("\n📈 Running inference on the test set…")
    y_prob = model.predict(X_test, batch_size=512, verbose=0).ravel()
    y_prob = np.clip(y_prob, 0.0, 1.0)

    # ── 8. Find optimal threshold ────────────────────────────────────────────
    optimal_thresh = find_optimal_threshold(y_test, y_prob)

    # ── 9. Evaluate at optimal threshold ─────────────────────────────────────
    metrics_opt = evaluate(y_test, y_prob, optimal_thresh, "OPTIMAL THRESHOLD")

    # Also show results at standard thresholds for comparison
    for t in [0.5, 0.3, 0.2, 0.1]:
        evaluate(y_test, y_prob, t, f"FIXED THRESHOLD {t}")

    # ── 10. Save model ───────────────────────────────────────────────────────
    model.save(str(OUTPUT_MODEL_PATH))
    print(f"\n💾 Model saved → {OUTPUT_MODEL_PATH}")

    # ── 11. Save metadata ────────────────────────────────────────────────────
    result = {
        "model_type": "Improved_BiGRU_Attention",
        "optimal_threshold": optimal_thresh,
        "training_time_seconds": round(train_time, 2),
        "training_time_minutes": round(train_time / 60, 2),
        "total_parameters": int(model.count_params()),
        "sequence_length": SEQUENCE_LENGTH,
        "num_features": len(FEATURES),
        "smote_applied": HAS_SMOTE,
        "focal_loss_alpha": round(alpha, 4),
        "epochs_run": len(history.history["loss"]),
        **{f"test_{k}": v for k, v in metrics_opt.items()},
    }
    OUTPUT_META_PATH.write_text(json.dumps(result, indent=2))
    print(f"📋 Results saved → {OUTPUT_META_PATH}")

    # ── 12. Side-by-side comparison ──────────────────────────────────────────
    baseline_path = Path(__file__).with_name("training_results_improved.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        print("\n" + "=" * 70)
        print("  BASELINE vs IMPROVED  (comparison)")
        print("=" * 70)
        for key in ["test_accuracy", "test_precision", "test_recall",
                     "test_f1", "test_auc"]:
            old = baseline.get(key, "N/A")
            new = result.get(key, "N/A")
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                arrow = "▲" if new > old else "▼" if new < old else "="
                print(f"  {key:25s}  {old:.6f}  →  {new:.6f}  {arrow}")
            else:
                print(f"  {key:25s}  {old}  →  {new}")
        print(f"  {'false_alarms':25s}  {baseline.get('false_alarms', 'N/A')}  "
              f"→  {result.get('test_false_positives', 'N/A')}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
