"""
Flash Crash Prediction — REST API Server
==========================================
Flask API that wraps the trained GRU models for the HTML/CSS/JS frontend.
Endpoints:
    GET  /api/models              — list available models
    POST /api/predict             — single-ticker prediction
    POST /api/portfolio           — multi-ticker portfolio scan
    POST /api/timeline            — rolling risk timeline
"""

import json, sys, traceback, time, random, hashlib, os
from pathlib import Path

import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ── custom layer registration ────────────────────────────────────────────────
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
try:
    from custom_layers import Attention, TemporalAttention
    CUSTOM_OBJECTS = {"Attention": Attention, "TemporalAttention": TemporalAttention}
except ImportError:
    CUSTOM_OBJECTS = {}

# ── app ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ── TradingView data via nologin method ──────────────────────────────────────
print("  TradingView: using nologin method for data access.")

# ── model registry ───────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    "msa_gru_best.keras",
    "improved_flash_crash_model.keras",
    "improved_minute_model.keras",
    "best_gru_model_improved.h5",
    "gru_model_final_improved.h5",
    "flash_crash_model.keras",
]

_loaded_models: dict[str, object] = {}


def _discover():
    return [c for c in MODEL_CANDIDATES if (MODEL_DIR / c).exists()]


def _get_model(name: str):
    if name not in _loaded_models:
        path = MODEL_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {name}")
        _loaded_models[name] = load_model(str(path), compile=False,
                                          custom_objects=CUSTOM_OBJECTS)
    return _loaded_models[name]


def _model_sig(model) -> tuple[int, int]:
    s = model.input_shape
    return int(s[1]), int(s[2])


# ── feature engineering ──────────────────────────────────────────────────────

FEATURES_14 = [
    "Open", "High", "Low", "Close", "Volume", "VWAP", "return",
    "volatility", "momentum", "volume_change", "vwap_diff",
    "high_low_spread", "open_close_return", "turnover_change",
]

FEATURES_10 = [
    "return", "log_return",
    "volatility_5", "volatility_10", "volatility_20",
    "momentum_5", "momentum_10",
    "high_low_spread", "open_close_return", "price_acceleration",
]

FEATURES_5 = [
    "return", "volume_change", "volatility_5", "volatility_10", "momentum_5",
]


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to title-cased OHLCV."""
    rename_map = {
        c: {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
             "date": "Date"}.get(str(c).strip().lower(), str(c).strip())
        for c in df.columns
    }
    return df.rename(columns=rename_map)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    o = _normalize_columns(_flatten(df.copy()))
    o["return"] = o["Close"].pct_change()             # matches train_minute.py
    o["log_return"] = np.log(o["Close"] / o["Close"].shift(1))
    o["volume_change"] = o["Volume"].pct_change()
    o["volatility_5"] = o["return"].rolling(5).std()
    o["volatility_10"] = o["return"].rolling(10).std()
    o["volatility_20"] = o["return"].rolling(20).std()
    o["volatility"] = o["volatility_10"]
    o["momentum_5"] = o["Close"].pct_change(5)
    o["momentum_10"] = o["Close"].pct_change(10)
    o["momentum"] = o["Close"] - o["Close"].shift(5)
    o["VWAP"] = (o["High"] + o["Low"] + o["Close"]) / 3.0
    o["vwap_diff"] = (o["Close"] - o["VWAP"]) / o["VWAP"].replace(0, np.nan)
    o["high_low_spread"] = (o["High"] - o["Low"]) / o["Close"].replace(0, np.nan)
    o["open_close_return"] = (o["Close"] - o["Open"]) / o["Open"].replace(0, np.nan)
    o["turnover_change"] = (o["Close"] * o["Volume"]).pct_change()
    o["price_acceleration"] = o["return"].diff()
    return o


def _pick_features(n_feat: int) -> list[str]:
    if n_feat == 14:
        return FEATURES_14
    elif n_feat == 10:
        return FEATURES_10
    elif n_feat == 5:
        return FEATURES_5
    return FEATURES_14[:n_feat]


def _build_seq(df, timesteps, n_feat):
    eng = _engineer(df)
    feats = _pick_features(n_feat)
    ff = eng[feats].copy()
    # Replace inf first, then forward-fill, then zero-fill any remaining NaN
    ff = ff.replace([np.inf, -np.inf], np.nan)
    ff = ff.ffill().bfill().fillna(0)
    if len(ff) < timesteps:
        raise ValueError(f"Need {timesteps} rows, got {len(ff)}")
    # Scale all features — models were trained on StandardScaler'd data
    sc = StandardScaler()
    ff = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)
    seq = np.expand_dims(ff.tail(timesteps).to_numpy(np.float32), 0)
    return seq, eng


def _demo_freq(interval: str) -> str:
    return {
        "1m": "min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "h",
        "1d": "B",
    }.get(interval, "B")


def _period_rows(period: str, interval: str) -> int:
    trading_days = {
        "1d": 1,
        "5d": 5,
        "7d": 7,
        "1mo": 22,
        "3mo": 66,
        "6mo": 132,
        "1y": 252,
        "2y": 504,
    }.get(period, 132)

    if interval in {"1m", "5m", "15m", "30m"}:
        minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}[interval]
        return max(60, (trading_days * 390) // minutes)
    if interval in {"60m", "1h"}:
        return max(30, trading_days * 6)
    return max(30, trading_days)


def _generate_demo_data(period: str, interval: str, seed: int = 42) -> pd.DataFrame:
    rows = _period_rows(period, interval)
    freq = _demo_freq(interval)
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq=freq)
    ret = rng.normal(0, 0.004 if interval in {"60m", "1h"} else 0.01, rows)
    close = 2000.0 * np.cumprod(1 + ret)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, rows))
    volume = rng.integers(500_000, 5_000_000, rows).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=ts,
    )


ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_STOCK_DIR = ROOT_DIR / "stk"

YF_CACHE_DIR = ROOT_DIR / ".yfinance_cache"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
YF_CACHE_TTL_SECONDS = 12 * 60 * 60  # 12 hours


def _cache_file_for(ticker: str, period: str, interval: str) -> Path:
    key = f"{ticker}|{period}|{interval}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return YF_CACHE_DIR / f"{digest}.csv"


def _write_disk_cache(df: pd.DataFrame, cache_fp: Path) -> None:
    try:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        out.index.name = "Date"
        out = out.reset_index()
        tmp = cache_fp.with_suffix(".tmp.csv")
        out.to_csv(tmp, index=False)
        tmp.replace(cache_fp)
    except Exception:
        return


def _read_disk_cache(cache_fp: Path) -> pd.DataFrame:
    df = pd.read_csv(cache_fp)
    # expected: Date column exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        # fallback: first column is datetime
        dt_col = df.columns[0]
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col)
    return df


def _load_local_daily_csv(ticker: str, period: str) -> pd.DataFrame | None:
    """
    Load OHLCV from local `stk/<SYMBOL>.csv` (offline).
    This is far more reliable than Yahoo for daily bars.
    """
    if not LOCAL_STOCK_DIR.exists():
        return None

    base = (ticker or "").strip().upper()
    if "." in base:
        base = base.split(".", 1)[0]
    base = base.replace("-", "")

    candidates = [LOCAL_STOCK_DIR / f"{base}.csv"]
    fp2 = LOCAL_STOCK_DIR / f"{base.replace('-', '')}.csv"
    if fp2 not in candidates:
        candidates.append(fp2)

    fp = next((c for c in candidates if c.exists()), None)
    if fp is None:
        return None

    try:
        df = pd.read_csv(fp)
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None

        # Keep only the app-required columns.
        df = df[["Date", *required]].rename(columns={"Date": "Date"})
        df = df.set_index("Date")
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Volume"] = df["Volume"].fillna(0.0)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        tail_n_map = {
            "1d": 1,
            "5d": 5,
            "7d": 7,
            "1mo": 22,
            "3mo": 66,
            "6mo": 132,
            "1y": 252,
            "2y": 504,
        }
        tail_n = tail_n_map.get(period, 132)
        df = df.tail(max(1, int(tail_n)))

        return _flatten(df)
    except Exception:
        return None


def _fetch(ticker, period="6mo", interval="1d", max_retries: int = 3):
    try:
        # nologin method — TvDatafeed() defaults to anonymous/nologin access
        tv = TvDatafeed()
    except Exception as e:
        print(f"Warning: could not initialize tvDatafeed: {e}")
        return _generate_demo_data(period, interval), True
        
    # Strip `.NS` for TradingView (e.g. RELIANCE.NS -> RELIANCE)
    base = (ticker or "").strip().upper()
    if "." in base:
        base = base.split(".")[0]
    
    interval_map = {
        "1m": Interval.in_1_minute,
        "5m": Interval.in_5_minute,
        "15m": Interval.in_15_minute,
        "30m": Interval.in_30_minute,
        "1h": Interval.in_1_hour,
        "60m": Interval.in_1_hour,
        "1d": Interval.in_daily,
        "1wk": Interval.in_weekly,
        "1mo": Interval.in_monthly
    }
    tv_interval = interval_map.get(interval, Interval.in_daily)
    
    last_err = ""
    for attempt in range(max_retries):
        try:
            # Fetch maximum history buffer available on free tier to ensure features compute nicely
            df = tv.get_hist(symbol=base, exchange='NSE', interval=tv_interval, n_bars=5000)
            
            if df is not None and not df.empty:
                # Map to expected yfinance style format
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                })
                if "symbol" in df.columns:
                    df = df.drop(columns=["symbol"])
                
                # Format index to ensure consistency
                df.index = pd.to_datetime(df.index)
                
                return df, False
            last_err = "Empty response from TradingView"
        except Exception as exc:
            last_err = str(exc)
        
        time.sleep(1)

    # Fallback to demo data if TradingView fails
    return _generate_demo_data(period, interval), True


def _risk_band(prob, threshold=0.20):
    if prob >= threshold:
        return "HIGH RISK"
    elif prob >= threshold * 0.65:
        return "ELEVATED"
    return "STABLE"


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    models = _discover()
    result = []
    for name in models:
        try:
            m = _get_model(name)
            ts, nf = _model_sig(m)
            result.append({"name": name, "timesteps": ts, "features": nf,
                           "params": int(m.count_params())})
        except Exception:
            pass
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        ticker = data.get("ticker", "RELIANCE.NS")
        model_name = data.get("model", _discover()[0])
        period = data.get("period", "6mo")
        interval = data.get("interval", "1d")
        threshold = float(data.get("threshold", 0.20))

        model = _get_model(model_name)
        ts, nf = _model_sig(model)

        df, is_demo = _fetch(ticker, period, interval)
        seq, eng = _build_seq(df, ts, nf)

        prob = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
        band = _risk_band(prob, threshold)

        # OHLC data for chart (last 60 bars)
        chart_df = _flatten(df).tail(60)
        ohlc = []
        for idx, row in chart_df.iterrows():
            ohlc.append({
                "date": str(idx),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
            })

        return jsonify({
            "ticker": ticker,
            "model": model_name,
            "probability": round(prob, 6),
            "risk_pct": round(prob * 100, 2),
            "band": band,
            "threshold": threshold,
            "timesteps": ts,
            "features": nf,
            "ohlc": ohlc,
            "latest_close": round(float(df["Close"].iloc[-1]), 2),
            "latest_date": str(df.index[-1])[:10],
            "source": "demo" if is_demo else "live",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/portfolio", methods=["POST"])
def portfolio():
    try:
        data = request.json
        tickers = data.get("tickers", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
        model_name = data.get("model", _discover()[0])
        period = data.get("period", "6mo")
        interval = data.get("interval", "1d")
        threshold = float(data.get("threshold", 0.20))

        model = _get_model(model_name)
        ts, nf = _model_sig(model)

        results = []
        for t in tickers:
            try:
                df, is_demo = _fetch(t, period, interval)
                seq, _ = _build_seq(df, ts, nf)
                prob = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
                results.append({
                    "ticker": t,
                    "probability": round(prob, 6),
                    "risk_pct": round(prob * 100, 2),
                    "band": _risk_band(prob, threshold),
                    "latest_close": round(float(df["Close"].iloc[-1]), 2),
                    "source": "demo" if is_demo else "live",
                })
            except Exception as e:
                results.append({"ticker": t, "error": str(e)})

        results.sort(key=lambda x: x.get("probability", 0), reverse=True)
        return jsonify({"results": results, "model": model_name, "threshold": threshold})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# ── Crash Replay definitions ─────────────────────────────────────────────────
CRASH_DATA_DIR = Path(__file__).resolve().parent.parent / "nifty 50 index minute data"

CRASH_EVENTS = [
    {"id": "nifty100_jun4", "label": "June 4, 2024 — Election Results (NIFTY 100)",
     "file": "NIFTY 100_minute.csv", "date": "2024-06-04",
     "start": "09:15", "end": "12:00", "crash_time": "10:35"},
    {"id": "infra_jun4", "label": "June 4, 2024 — Infra Shock (NIFTY INFRA)",
     "file": "NIFTY INFRA_minute.csv", "date": "2024-06-04",
     "start": "09:15", "end": "12:00", "crash_time": "10:33"},
    {"id": "auto_jun4", "label": "June 4, 2024 — Auto Selloff (NIFTY AUTO)",
     "file": "NIFTY AUTO_minute.csv", "date": "2024-06-04",
     "start": "10:00", "end": "13:00", "crash_time": "11:51"},
]


@app.route("/api/crash-events", methods=["GET"])
def list_crash_events():
    return jsonify(CRASH_EVENTS)


@app.route("/api/crash-replay", methods=["POST"])
def crash_replay():
    try:
        data = request.json
        event_id = data.get("event_id", CRASH_EVENTS[0]["id"])
        model_name = data.get("model", _discover()[0])

        event = next((e for e in CRASH_EVENTS if e["id"] == event_id), None)
        if not event:
            return jsonify({"error": f"Unknown crash event: {event_id}"}), 400

        csv_path = CRASH_DATA_DIR / event["file"]
        if not csv_path.exists():
            return jsonify({"error": f"Data file not found: {event['file']}"}), 404

        model = _get_model(model_name)
        ts, nf = _model_sig(model)
        feats = _pick_features(nf)

        df = pd.read_csv(csv_path)
        df = _normalize_columns(df)
        df["Date"] = pd.to_datetime(df["Date"])

        start_dt = pd.to_datetime(f"{event['date']} {event['start']}")
        end_dt = pd.to_datetime(f"{event['date']} {event['end']}")
        window = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
        window = window.set_index("Date")

        eng = _engineer(window)
        ff = eng[feats].dropna()
        ff = ff.replace([np.inf, -np.inf], np.nan).fillna(0)

        sc = StandardScaler()
        ff = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)
        arr = ff.to_numpy(np.float32)

        risk_points = []
        for i in range(ts, len(arr)):
            seq = np.expand_dims(arr[i - ts:i], 0)
            p = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
            risk_points.append({
                "date": str(ff.index[i - 1]),
                "risk": round(p * 100, 2),
            })

        # OHLC for chart
        ohlc = []
        for idx, row in window.iterrows():
            ohlc.append({
                "date": str(idx),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
            })

        return jsonify({
            "event": event,
            "model": model_name,
            "risk_points": risk_points,
            "ohlc": ohlc,
            "peak_risk": max(p["risk"] for p in risk_points) if risk_points else 0,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        model_name = request.form.get("model", _discover()[0])
        threshold = float(request.form.get("threshold", 0.20))

        model = _get_model(model_name)
        ts, nf = _model_sig(model)

        import io
        df = pd.read_csv(io.BytesIO(f.read()))
        df = _normalize_columns(df)

        seq, eng = _build_seq(df, ts, nf)
        prob = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
        band = _risk_band(prob, threshold)

        # Feature snapshot — use same NaN handling as _build_seq
        feats = _pick_features(nf)
        feat_df = eng[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        if len(feat_df) > 0:
            feat_vals = feat_df.tail(1).iloc[0].to_dict()
            feat_snapshot = {k: round(float(v), 6) for k, v in feat_vals.items()}
        else:
            feat_snapshot = {}

        return jsonify({
            "probability": round(prob, 6),
            "risk_pct": round(prob * 100, 2),
            "band": band,
            "threshold": threshold,
            "model": model_name,
            "rows_loaded": len(df),
            "features": feat_snapshot,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/timeline", methods=["POST"])
def timeline():
    try:
        data = request.json
        ticker = data.get("ticker", "RELIANCE.NS")
        model_name = data.get("model", _discover()[0])
        period = data.get("period", "1y")
        interval = data.get("interval", "1d")

        model = _get_model(model_name)
        ts, nf = _model_sig(model)
        feats = _pick_features(nf)

        df, is_demo = _fetch(ticker, period, interval)
        eng = _engineer(df)
        ff = eng[feats].dropna()
        ff = ff.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale all features — models were trained on StandardScaler'd data
        sc = StandardScaler()
        ff = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)

        arr = ff.to_numpy(np.float32)
        step = max(1, len(arr) // 200)  # ~200 points max

        points = []
        for i in range(ts, len(arr), step):
            seq = np.expand_dims(arr[i - ts:i], 0)
            p = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
            points.append({
                "date": str(ff.index[i - 1])[:10],
                "risk": round(p * 100, 2),
            })

        return jsonify({
            "ticker": ticker,
            "model": model_name,
            "points": points,
            "source": "demo" if is_demo else "live",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/stream", methods=["GET"])
def stream_api():
    try:
        ticker = request.args.get("ticker", "RELIANCE.NS").strip().upper()
        # upstox expects format NSE_EQ|RELIANCE
        base = ticker.replace(".NS", "")
        inst_key = f"NSE_EQ|{base}"
        
        token = request.args.get("token", "").strip()
        model_name = request.args.get("model", _discover()[0])
        threshold = float(request.args.get("threshold", 0.20))
        
        model = _get_model(model_name)
        ts, nf = _model_sig(model)
        
        def event_stream():
            import upstox_client
            from datetime import datetime, timedelta
            from flask import json
            
            conf = upstox_client.Configuration()
            conf.access_token = token
            api_client = upstox_client.ApiClient(conf)
            history_api = upstox_client.HistoryApi(api_client)
            quote_api = upstox_client.MarketQuoteApi(api_client)
            
            # Fetch 30 days of 1-minute historical data for feature generation
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            try:
                hist_res = history_api.get_historical_candle_data1(
                    instrument_key=inst_key,
                    interval="1minute",  # Enum allowed by Upstox SDK
                    to_date=end_date,
                    from_date=start_date,
                    api_version="2.0"
                )
                
                # Upstox returns data newest first usually, let's parse safely
                candles = hist_res.data.candles
                # Candles are [timestamp, open, high, low, close, vol, oi]
                # Sort chronological: oldest first
                candles.sort(key=lambda x: x[0])
                
                records = []
                for c in candles:
                    dt = pd.to_datetime(c[0])
                    records.append({"Date": dt, "Open": float(c[1]), "High": float(c[2]), "Low": float(c[3]), "Close": float(c[4]), "Volume": float(c[5])})
                    
                df_hist = pd.DataFrame(records).set_index("Date")
                
            except Exception as e:
                yield f"data: {json.dumps({'error': f'Failed to fetch history: {str(e)}'})}\n\n"
                return
                
            while True:
                try:
                    # 1. Fetch live quote
                    q_res = quote_api.get_market_quote(symbol=inst_key, api_version="2.0")
                    quote = q_res.data.get(inst_key)
                    if not quote:
                        time.sleep(2)
                        continue
                        
                    ltp = float(quote.last_price)
                    vol = float(quote.volume)
                    
                    # 2. Append/Update current live tick array as the newest candle
                    now = pd.to_datetime("now", utc=True)
                    # We simply append a new row for the LTP to evaluate instantaneous risk
                    live_row = pd.DataFrame([{"Date": now, "Open": ltp, "High": ltp, "Low": ltp, "Close": ltp, "Volume": vol}]).set_index("Date")
                    df_live = pd.concat([df_hist, live_row])
                    
                    # 3. Compute risk
                    seq, eng = _build_seq(df_live, ts, nf)
                    prob = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
                    band = _risk_band(prob, threshold)
                    
                    # Feature snapshot
                    feats = _pick_features(nf)
                    feat_df = eng[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
                    feat_snapshot = {k: round(float(v), 4) for k, v in feat_df.iloc[-1].to_dict().items()}
                    
                    # Construct chart candles (last 60)
                    chart_df = df_live.tail(60)
                    ohlc = []
                    for idx, row in chart_df.iterrows():
                        ohlc.append({
                            "ts": str(idx),
                            "open": round(float(row["Open"]), 2),
                            "high": round(float(row["High"]), 2),
                            "low": round(float(row["Low"]), 2),
                            "close": round(float(row["Close"]), 2),
                            "volume": int(row["Volume"]),
                        })
                    
                    payload = {
                        "ticker": ticker,
                        "risk_pct": round(prob * 100, 2),
                        "probability": round(prob, 6),
                        "band": band,
                        "latest_close": ltp,
                        "features": feat_snapshot,
                        "candles": ohlc,
                        "status": "streaming"
                    }
                    
                    yield f"data: {json.dumps(payload)}\n\n"
                    
                except Exception as e:
                    print(f"Stream error: {e}")
                    # keep alive
                    yield f": keep-alive\n\n"
                    
                time.sleep(2) # Stream interval
                
        from flask import Response
        return Response(event_stream(), mimetype="text/event-stream")
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# ── Upstox Proxy Endpoints (bypass browser CORS) ───────────────────────────
import urllib.request, urllib.parse

UPSTOX_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
UPSTOX_CANDLE_URL = "https://api.upstox.com/v2/historical-candle"
UPSTOX_API_KEY = "0f744985-4514-4274-84a5-037388d02bdd"
UPSTOX_API_SECRET = "pptigfbzw6"
UPSTOX_REDIRECT_URL = "http://localhost:3000/"

# Store token in memory (server-side session)
_upstox_token = {"value": None}


@app.route("/api/upstox-token", methods=["POST"])
def upstox_token():
    """Exchange OAuth code for access token (server-side to bypass CORS)."""
    try:
        import urllib.request, urllib.parse
        data = request.json or {}
        code = data.get("code", "").strip()
        redirect_uri = data.get("redirect_uri", UPSTOX_REDIRECT_URL).strip()
        if not code:
            return jsonify({"error": "Missing code"}), 400

        body = urllib.parse.urlencode({
            "code": code,
            "client_id": UPSTOX_API_KEY,
            "client_secret": UPSTOX_API_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }).encode()

        req = urllib.request.Request(
            UPSTOX_TOKEN_URL, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())

        if "access_token" in result:
            _upstox_token["value"] = result["access_token"]
            return jsonify({"access_token": result["access_token"], "status": "ok"})
        else:
            return jsonify({"error": result.get("message", "Token exchange failed"), "detail": result}), 400

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return jsonify({"error": f"Upstox returned {e.code}", "detail": body}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/upstox-candles", methods=["GET"])
def upstox_candles():
    """Proxy Upstox historical-candle API to avoid CORS (server-side fetch)."""
    try:
        import urllib.request
        sym = request.args.get("sym", "RELIANCE").strip().upper()
        interval = request.args.get("interval", "5minute")
        to_date = request.args.get("to", "")
        from_date = request.args.get("from", "")
        token = request.headers.get("X-Upstox-Token") or _upstox_token.get("value") or ""

        if not to_date or not from_date:
            from datetime import datetime, timedelta
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        inst_key = f"NSE_EQ|{sym}"
        url = f"{UPSTOX_CANDLE_URL}/{urllib.parse.quote(inst_key)}/{interval}/{to_date}/{from_date}"

        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

        return jsonify(data)

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return jsonify({"error": f"Upstox {e.code}", "detail": body}), e.code
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    print("=" * 60)
    print("  Flash Crash Prediction — API Server")
    print("=" * 60)
    print(f"  Models found: {_discover()}")
    print(f"  Frontend: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
