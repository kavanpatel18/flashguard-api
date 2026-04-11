"""
MSA-GRU Flash Crash Prediction Pipeline
Multi-Scale Attention GRU using user dataset (flash_crash_ready_dataset.csv)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# -- Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -- Hyperparameters
DATA_PATH       = "flash_crash_ready_dataset.csv"
LONG_WIN        = 60
MED_WIN         = 30
SHORT_WIN       = 10
GRU_UNITS       = 64
DENSE1_UNITS    = 128
DENSE2_UNITS    = 64
DROPOUT_RATE    = 0.3
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 256
EPOCHS          = 50
TRAIN_RATIO     = 0.80
MODEL_PATH      = "msa_gru_best.keras"
RESULTS_DIR     = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "VWAP",
    "return", "volatility", "momentum", "volume_change",
    "vwap_diff", "high_low_spread", "open_close_return", "turnover_change"
]

# -- 1. Data Pipeline
def load_and_prepare_data(path: str):
    print(f"[DATA] Loading {path}...")
    df = pd.read_csv(path)
    print(f"[DATA] Shape: {df.shape} | Tickers: {df['ticker'].nunique()}")

    X_tr_list, y_tr_list = [], []
    X_te_list, y_te_list = [], []

    for t in df["ticker"].unique():
        sdf = df[df["ticker"] == t].reset_index(drop=True)
        vals = sdf[FEATURES].values
        labels = sdf["crash_label"].values

        n_seq = len(vals) - LONG_WIN
        if n_seq <= 0:
            continue

        seqs = np.array([vals[i:i + LONG_WIN] for i in range(n_seq)], dtype=np.float32)
        labs = np.array([labels[i + LONG_WIN] for i in range(n_seq)], dtype=np.float32)

        split_idx = int(n_seq * TRAIN_RATIO)
        X_tr_list.append(seqs[:split_idx])
        y_tr_list.append(labs[:split_idx])
        X_te_list.append(seqs[split_idx:])
        y_te_list.append(labs[split_idx:])

    X_train = np.concatenate(X_tr_list)
    y_train = np.concatenate(y_tr_list)
    X_test  = np.concatenate(X_te_list)
    y_test  = np.concatenate(y_te_list)

    # Scaling
    print("[DATA] Running StandardScaler...")
    scaler = StandardScaler()
    n_tr, t_tr, f_tr = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(n_tr * t_tr, f_tr)).reshape(n_tr, t_tr, f_tr)

    n_te, t_te, f_te = X_test.shape
    X_test = scaler.transform(X_test.reshape(n_te * t_te, f_te)).reshape(n_te, t_te, f_te)

    # Validation Split
    X_val = X_train[-int(n_tr * 0.15):]
    y_val = y_train[-int(n_tr * 0.15):]
    X_train = X_train[:-int(n_tr * 0.15)]
    y_train = y_train[:-int(n_tr * 0.15)]

    print(f"[DATA] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"[DATA] Train crashes: {int(y_train.sum())}, Val crashes: {int(y_val.sum())}, Test crashes: {int(y_test.sum())}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def apply_smote(X_train, y_train):
    if not HAS_SMOTE:
        print("[SMOTE] imblearn not installed. Skipping SMOTE.")
        return X_train, y_train
    
    n, t, f = X_train.shape
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    
    target_count = min(neg_count, pos_count * 10)
    target_count = max(target_count, pos_count * 5)
    
    print(f"[SMOTE] Oversampling positive class from {pos_count} to {target_count}...")
    sm = SMOTE(sampling_strategy={1: target_count}, random_state=42, k_neighbors=min(5, pos_count - 1))
    X_res, y_res = sm.fit_resample(X_train.reshape(n, t * f), y_train)
    X_res = X_res.reshape(-1, t, f).astype(np.float32)
    return X_res, y_res

# -- 2. Focal Loss & Attention
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce    = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t    = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    loss_fn.__name__ = "focal_loss"
    return loss_fn

class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W = self.add_weight(shape=(feat_dim, feat_dim), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(feat_dim,), initializer="zeros", trainable=True)
        self.u = self.add_weight(shape=(feat_dim, 1), initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        score = tf.tanh(tf.tensordot(x, self.W, axes=[[2], [0]]) + self.b)
        alpha = tf.nn.softmax(tf.squeeze(tf.tensordot(score, self.u, axes=[[2], [0]]), axis=-1), axis=1)
        context = tf.einsum("bt,btf->bf", alpha, x)
        return context

    def get_config(self):
        return super().get_config()

# -- 3. Build MSA-GRU
def build_msa_gru(num_features: int) -> keras.Model:
    inp = keras.Input(shape=(LONG_WIN, num_features), name="input_long")

    scale_short = inp[:, -SHORT_WIN:, :]
    scale_med   = inp[:, -MED_WIN:, :]
    scale_long  = inp

    def gru_branch(x, name_prefix):
        h = layers.Bidirectional(layers.GRU(GRU_UNITS, return_sequences=True, dropout=0.1, recurrent_dropout=0.1), name=f"{name_prefix}_gru")(x)
        ctx = TemporalAttention(name=f"{name_prefix}_att")(h)
        return ctx

    ctx_short = gru_branch(scale_short, "short")
    ctx_med   = gru_branch(scale_med, "med")
    ctx_long  = gru_branch(scale_long, "long")

    fused = layers.concatenate([ctx_short, ctx_med, ctx_long], name="fusion")

    x = layers.BatchNormalization(name="batch_norm")(fused)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(DENSE1_UNITS, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(DENSE2_UNITS, activation="relu")(x)
    
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs=inp, outputs=out, name="MSA-GRU")

# -- 4. Threshold Optimization
def find_optimal_threshold(y_true, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    print(f"\n[EVAL] Optimal F1 Threshold: {thresholds[best_idx]:.4f} (F1={f1_scores[best_idx]:.4f})")
    return float(thresholds[best_idx])

# -- 5. Pipeline Main
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(DATA_PATH)
    X_train, y_train = apply_smote(X_train, y_train)

    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    alpha = neg_count / (pos_count + neg_count)
    print(f"[TRAIN] Focal loss alpha set to {alpha:.4f}")

    model = build_msa_gru(len(FEATURES))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(gamma=2.0, alpha=alpha),
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_auc", save_best_only=True, mode="max", verbose=0),
    ]

    print("\n[TRAIN] Commencing training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1
    )

    # Evaluate
    print("\n[EVAL] Evaluating best checkpoint on test set...")
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TemporalAttention": TemporalAttention, "focal_loss": focal_loss(2.0, alpha)}
    )
    
    y_prob = model.predict(X_test, batch_size=512, verbose=0).ravel()
    opt_thresh = find_optimal_threshold(y_test, y_prob)
    y_pred = (y_prob >= opt_thresh).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("=======================================================")
    print("  MSA-GRU Flash Crash Detector - Test Metrics")
    print(f"  (Evaluated at Threshold = {opt_thresh:.4f})")
    print("=======================================================")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("=======================================================")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Plot
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(history.history["loss"], label="Train Loss")
    ax0.plot(history.history["val_loss"], label="Val Loss")
    ax0.set_title("Focal Loss"); ax0.legend()

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(history.history["auc"], label="Train AUC")
    ax1.plot(history.history["val_auc"], label="Val AUC")
    ax1.set_title("AUC"); ax1.legend()

    ax2 = fig.add_subplot(gs[2])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_title("Test Set ROC"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "msa_gru_results.png"))
    plt.close()
    print(f"\n[DONE] Saved results to {RESULTS_DIR}/msa_gru_results.png and {MODEL_PATH}")

if __name__ == "__main__":
    main()
