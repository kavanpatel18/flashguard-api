"""
FlashGuard v4 — Upstox-First Real-Time API Server
===================================================
• Primary data source: Upstox v2 REST API (hardcoded token or per-request)
• Fallback: yFinance (delayed)  →  demo data
• Model: improved_minute_model.keras  (30 timesteps × 10 features)
• Attention layer weights: W=(u,u)  b=(u,)  V=(u,)  [flat]

Run:
    pip install flask flask-cors numpy pandas requests scikit-learn tensorflow "yfinance==0.2.38"
    python api_server.py
    → http://localhost:5000
"""

import gzip, io, sys, traceback, time, random, json, threading
from datetime import date, timedelta, datetime
from pathlib import Path

import numpy  as np
import pandas as pd
import requests

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

# ── Keras ─────────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    import keras
    from keras.models import load_model as _keras_load
    KERAS_OK = True
    print(f"  TensorFlow {tf.__version__} ready")
except ImportError:
    KERAS_OK = False
    print("  WARNING: TensorFlow not installed")

if KERAS_OK:
    @keras.saving.register_keras_serializable(package="custom_layers")
    class Attention(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            u = input_shape[-1]
            self.W = self.add_weight(shape=(u,u), initializer="glorot_uniform", name="att_weight")
            self.b = self.add_weight(shape=(u,),  initializer="zeros",          name="att_bias")
            self.V = self.add_weight(shape=(u,),  initializer="glorot_uniform", name="att_v")
            super().build(input_shape)
        def call(self, x):
            s = tf.nn.tanh(tf.tensordot(x, self.W, axes=[[2],[0]]) + self.b)
            w = tf.nn.softmax(tf.reduce_sum(s * self.V, axis=-1, keepdims=True), axis=1)
            return tf.reduce_sum(x * w, axis=1)
        def get_config(self):
            return super().get_config()
    _CUSTOM = {"Attention": Attention}
else:
    _CUSTOM = {}

# ── Paths & Flask ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRONTEND = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
CORS(app)

# ── Upstox config ─────────────────────────────────────────────────────────────
UPSTOX_BASE = "https://api.upstox.com/v2"

# Default token — can be overridden per-request
DEFAULT_TOKEN = ""   # User pastes token in the UI; this is the hardcoded fallback

# ── Dynamic Upstox instrument master ──────────────────────────────────────────
INSTRUMENT_LIST_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
_instrument_cache: list[dict] = []          # [{symbol, name, instrument_key, segment}, …]
_instrument_lock = threading.Lock()
_instrument_loaded = False


def _load_instruments():
    """Download & cache the Upstox master instrument file."""
    global _instrument_cache, _instrument_loaded
    with _instrument_lock:
        if _instrument_loaded:
            return
        try:
            print("  Downloading Upstox instrument master…")
            resp = requests.get(INSTRUMENT_LIST_URL, timeout=30)
            resp.raise_for_status()
            raw = gzip.decompress(resp.content)
            instruments = json.loads(raw)
            filtered = []
            for inst in instruments:
                seg = inst.get("segment", "")
                if seg not in ("NSE_EQ", "NSE_INDEX", "BSE_INDEX"):
                    continue
                sym  = inst.get("trading_symbol", "").strip()
                name = inst.get("name", "").strip()
                ikey = inst.get("instrument_key", "").strip()
                if sym and ikey:
                    filtered.append({"symbol": sym, "name": name,
                                     "instrument_key": ikey, "segment": seg})
            _instrument_cache = sorted(filtered, key=lambda x: x["symbol"])
            _instrument_loaded = True
            print(f"  Loaded {len(_instrument_cache)} instruments")
        except Exception as e:
            print(f"  Failed to load instruments: {e}")


def _search_instruments(query: str, limit: int = 15) -> list[dict]:
    """Search cached instruments by symbol/name (case-insensitive)."""
    if not _instrument_loaded:
        _load_instruments()
    q = query.upper().strip()
    if not q:
        return []
    prefix, substr = [], []
    for inst in _instrument_cache:
        sym_up  = inst["symbol"].upper()
        name_up = inst["name"].upper()
        if sym_up.startswith(q):
            prefix.append(inst)
        elif q in sym_up or q in name_up:
            substr.append(inst)
        if len(prefix) + len(substr) >= limit * 2:
            break
    return (prefix + substr)[:limit]


def _resolve_instrument_key(ticker: str, instrument_key: str | None = None) -> str | None:
    """Resolve an Upstox instrument_key for a ticker.
    Uses explicit key if provided, else looks up via cached master file."""
    if instrument_key:
        return instrument_key
    sym = ticker.upper().replace(".NS", "").strip()
    if not _instrument_loaded:
        _load_instruments()
    for inst in _instrument_cache:
        if inst["symbol"].upper() == sym:
            return inst["instrument_key"]
    return None

# Upstox interval map
IV_MAP = {
    "1m":  "1minute",
    "5m":  "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h":  "60minute",
    "1d":  "day",
    "1wk": "week",
    "1mo": "month",
}

INTRADAY_IVS = ("1minute","5minute","15minute","30minute","60minute")

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_PRIORITY = [
    "improved_minute_model.keras",
    "improved_flash_crash_model.keras",
]
_loaded: dict = {}

def _discover():
    return [m for m in MODEL_PRIORITY if (BASE_DIR / m).exists()]

def _load(name: str):
    if name in _loaded:
        return _loaded[name]
    fp = BASE_DIR / name
    if not fp.exists():
        raise FileNotFoundError(f"Model not found: {name}")
    if not KERAS_OK:
        raise RuntimeError("TensorFlow not installed")
    m = _keras_load(str(fp), compile=False, custom_objects=_CUSTOM)
    _loaded[name] = m
    print(f"  ✓ Loaded {name}: input={m.input_shape}")
    return m

def _sig(model):
    s = model.input_shape
    return int(s[1]), int(s[2])

def _best():
    f = _discover()
    if not f: raise RuntimeError("No model files found next to api_server.py")
    return f[0]

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES_10 = [
    "return", "log_return",
    "volatility_5", "volatility_10", "volatility_20",
    "momentum_5", "momentum_10",
    "high_low_spread", "open_close_return", "price_acceleration",
]
FEATURES_14 = [
    "Open","High","Low","Close","Volume","VWAP","return",
    "volatility","momentum","volume_change","vwap_diff",
    "high_low_spread","open_close_return","turnover_change",
]
FEATURES_5 = ["return","volume_change","volatility_5","volatility_10","momentum_5"]

def _pick_features(n):
    if n == 10: return FEATURES_10
    if n == 14: return FEATURES_14
    if n == 5:  return FEATURES_5
    return FEATURES_10[:n]

def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _norm(df):
    m = {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume","date":"Date"}
    return df.rename(columns={c: m.get(str(c).strip().lower(), str(c).strip()) for c in df.columns})

def _engineer(df):
    o = _norm(_flatten(df.copy()))
    o["return"]             = o["Close"].pct_change()
    o["log_return"]         = np.log(o["Close"] / o["Close"].shift(1))
    o["volume_change"]      = o["Volume"].pct_change()
    o["volatility_5"]       = o["return"].rolling(5).std()
    o["volatility_10"]      = o["return"].rolling(10).std()
    o["volatility_20"]      = o["return"].rolling(20).std()
    o["volatility"]         = o["volatility_10"]
    o["momentum_5"]         = o["Close"].pct_change(5)
    o["momentum_10"]        = o["Close"].pct_change(10)
    o["momentum"]           = o["Close"] - o["Close"].shift(5)
    o["VWAP"]               = (o["High"] + o["Low"] + o["Close"]) / 3.0
    o["vwap_diff"]          = (o["Close"] - o["VWAP"]) / o["VWAP"].replace(0, np.nan)
    o["high_low_spread"]    = (o["High"] - o["Low"]) / o["Close"].replace(0, np.nan)
    o["open_close_return"]  = (o["Close"] - o["Open"]) / o["Open"].replace(0, np.nan)
    o["turnover_change"]    = (o["Close"] * o["Volume"]).pct_change()
    o["price_acceleration"] = o["return"].diff()
    return o

def _build_seq(df, timesteps, n_feat):
    eng   = _engineer(df)
    feats = _pick_features(n_feat)
    ff    = eng[feats].replace([np.inf,-np.inf], np.nan).bfill().ffill().fillna(0)
    if len(ff) < timesteps:
        raise ValueError(f"Need {timesteps} rows, only got {len(ff)}. Use a longer period.")
    sc  = StandardScaler()
    ff  = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)
    return np.expand_dims(ff.tail(timesteps).to_numpy(np.float32), 0), eng

# ── Upstox data fetcher (PRIMARY) ─────────────────────────────────────────────
def _upstox_headers(token=None):
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def _upstox_candles(instrument_key, interval, token, period="6mo"):
    """Fetch historical + intraday candles from Upstox v2."""
    enc = instrument_key.replace("|", "%7C")
    hdr = _upstox_headers(token)
    all_candles = []

    days_map = {"5d":7,"1mo":35,"3mo":95,"6mo":185,"1y":370,"2y":740,"5y":1825,"6y":2200,"10y":3650}

    days = days_map.get(period, 185)
    fd   = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    td_  = date.today().strftime("%Y-%m-%d")
    url  = f"{UPSTOX_BASE}/historical-candle/{enc}/{interval}/{td_}/{fd}"

    try:
        r = requests.get(url, headers=hdr, timeout=12)
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                all_candles = d.get("data", {}).get("candles", [])
    except Exception as e:
        print(f"  Upstox historical error: {e}")

    # Fetch today's intraday for all intraday intervals
    if interval in INTRADAY_IVS:
        try:
            url2 = f"{UPSTOX_BASE}/historical-candle/intraday/{enc}/{interval}"
            r2   = requests.get(url2, headers=hdr, timeout=8)
            if r2.status_code == 200:
                d2 = r2.json()
                if d2.get("status") == "success":
                    all_candles += d2.get("data", {}).get("candles", [])
        except Exception as e:
            print(f"  Upstox intraday error: {e}")

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles,
                      columns=["Datetime","Open","High","Low","Close","Volume","OI"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = (df.drop_duplicates("Datetime")
            .set_index("Datetime")
            .sort_index())
    return df[["Open","High","Low","Close","Volume"]].astype(float)


def _upstox_quote(instrument_key, token):
    """Get live LTP from Upstox market quote."""
    try:
        hdr = _upstox_headers(token)
        enc = instrument_key.replace("|", "%7C")
        url = f"{UPSTOX_BASE}/market-quote/quotes?instrument_key={enc}"
        r   = requests.get(url, headers=hdr, timeout=8)
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                data = d.get("data", {})
                key  = list(data.keys())[0] if data else None
                if key:
                    q = data[key]
                    return {
                        "ltp":        q.get("last_price", 0),
                        "open":       q.get("ohlc", {}).get("open", 0),
                        "high":       q.get("ohlc", {}).get("high", 0),
                        "low":        q.get("ohlc", {}).get("low", 0),
                        "prev_close": q.get("ohlc", {}).get("close", 0),
                        "volume":     q.get("volume", 0),
                        "change":     q.get("net_change", 0),
                        "change_pct": q.get("net_change", 0) / max(q.get("ohlc",{}).get("close",1),1) * 100,
                    }
    except Exception as e:
        print(f"  Upstox quote error: {e}")
    return None


def _market_overview(token):
    """Fetch live quotes for major indices and stocks."""
    tickers = {
        "NIFTY 50":   "NSE_INDEX|Nifty 50",
        "BANK NIFTY": "NSE_INDEX|Nifty Bank",
        "RELIANCE":   "NSE_EQ|INE002A01018",
        "TCS":        "NSE_EQ|INE467B01029",
        "HDFC BANK":  "NSE_EQ|INE040A01034",
        "INFOSYS":    "NSE_EQ|INE009A01021",
        "ICICI BANK": "NSE_EQ|INE090A01021",
        "SBI":        "NSE_EQ|INE062A01020",
    }
    results = []
    keys_encoded = ",".join(v.replace("|","%7C") for v in tickers.values())
    try:
        hdr = _upstox_headers(token)
        url = f"{UPSTOX_BASE}/market-quote/quotes?instrument_key={keys_encoded}"
        r   = requests.get(url, headers=hdr, timeout=10)
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                data = d.get("data", {})
                for name, ikey in tickers.items():
                    # Upstox returns key with | replaced by :
                    lookup = ikey.replace("|",":")
                    q = data.get(lookup) or data.get(ikey) or {}
                    if not q:
                        # Try matching by partial key
                        for dk in data.keys():
                            if ikey.split("|")[1] in dk:
                                q = data[dk]; break
                    ltp   = q.get("last_price", 0)
                    ohlc  = q.get("ohlc", {})
                    pclose= ohlc.get("close", ltp) or ltp
                    chg   = ltp - pclose
                    chgp  = (chg / pclose * 100) if pclose else 0
                    results.append({
                        "name":    name,
                        "ltp":     round(ltp, 2),
                        "change":  round(chg, 2),
                        "change_pct": round(chgp, 2),
                        "volume":  q.get("volume", 0),
                        "high":    round(ohlc.get("high", 0), 2),
                        "low":     round(ohlc.get("low", 0), 2),
                    })
    except Exception as e:
        print(f"  Market overview error: {e}")
    return results


# ── Demo data fallback ─────────────────────────────────────────────────────────
def _demo(period="6mo", interval="1d", seed=42):
    td = {"5d":5,"1mo":22,"3mo":66,"6mo":132,"1y":252,"2y":504,"5y":1260}.get(period,132)
    intraday_mins = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60}
    if interval in intraday_mins:
        mins = intraday_mins[interval]; rows = max(60,(td*390)//mins); freq=f"{mins}min"
    else:
        rows = max(30,td); freq="B"
    rng   = np.random.default_rng(seed)
    ts    = pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq=freq)
    ret   = rng.normal(0,0.008,rows)
    close = 2450.0 * np.cumprod(1+ret)
    open_ = np.r_[close[0],close[:-1]]*(1+rng.normal(0,0.001,rows))
    high  = np.maximum(open_,close)*(1+rng.uniform(0.001,0.01,rows))
    low   = np.minimum(open_,close)*(1-rng.uniform(0.001,0.01,rows))
    vol   = rng.integers(800_000,8_000_000,rows).astype(float)
    return pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Volume":vol},index=ts)


# ── Main fetch function ────────────────────────────────────────────────────────
def _fetch(ticker, period="6mo", interval="1d", token=None, instrument_key=None):
    """
    1. Try Upstox (if token provided)
    2. Try yFinance
    3. Fall back to demo data
    Returns (DataFrame, source_string)
    """
    clean = ticker.upper().replace(".NS","").strip()
    ikey  = _resolve_instrument_key(ticker, instrument_key)
    up_iv = IV_MAP.get(interval, "day")

    # ── 1. Upstox (historical candles are public — no token needed) ─────────
    if ikey:
        try:
            df = _upstox_candles(ikey, up_iv, token, period)
            if df is not None and len(df) > 5:
                print(f"  ✓ Upstox: {len(df)} candles for {ticker}")
                return df, "upstox"
            else:
                print(f"  Upstox returned empty for {ticker}")
        except Exception as e:
            print(f"  Upstox fetch error: {e}")

    # ── 2. yFinance ────────────────────────────────────────────────────────────
    try:
        import yfinance as yf
        yf_iv_map = {
            "1m":"1m","5m":"5m","15m":"15m","30m":"30m",
            "1h":"1h","1d":"1d","1wk":"1wk","1mo":"1mo"
        }
        yf_period_cap = {"1m":"5d","5m":"60d","15m":"60d","30m":"60d"}
        period_days  = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365,
                        "2y":730,"5y":1825,"6y":2190,"10y":3650,"60d":60,"7d":7}
        yf_iv  = yf_iv_map.get(interval, "1d")
        eff_p  = yf_period_cap.get(interval, period)
        ndays  = period_days.get(eff_p, 180)
        start  = (date.today() - timedelta(days=ndays)).strftime("%Y-%m-%d")
        end    = date.today().strftime("%Y-%m-%d")
        sym = ticker if ".NS" in ticker else (ticker + ".NS" if ticker not in ["NIFTY50","BANKNIFTY","SENSEX"] else "^NSEI")
        df = yf.Ticker(sym).history(start=start, end=end, interval=yf_iv)
        if df is not None and not df.empty:
            df = _flatten(df)
            # Normalise index to UTC-aware timestamps
            idx = pd.to_datetime(df.index)
            if idx.tzinfo is None or getattr(idx, 'tz', None) is None:
                try: idx = idx.tz_localize('UTC')
                except Exception: pass
            else:
                try: idx = idx.tz_convert('UTC')
                except Exception: pass
            df.index = idx
            print(f"  ✓ yFinance: {len(df)} rows for {ticker}")
            return df, "yfinance"
    except Exception as e:
        print(f"  yFinance error: {e}")

    # ── 3. Demo ────────────────────────────────────────────────────────────────
    print(f"  Using demo data for {ticker}")
    return _demo(period, interval), "demo"


def _risk_band(prob, threshold=0.20):
    if prob >= threshold:        return "HIGH RISK"
    if prob >= threshold * 0.65: return "ELEVATED"
    return "STABLE"


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    for d in (FRONTEND, BASE_DIR):
        if (d/"index.html").exists(): return send_from_directory(str(d),"index.html")
    return "index.html not found", 404

@app.route("/dashboard.html")
def dashboard():
    for d in (FRONTEND, BASE_DIR):
        if (d/"dashboard.html").exists(): return send_from_directory(str(d),"dashboard.html")
    return "dashboard.html not found", 404

@app.route("/<path:f>")
def static_files(f):
    for d in (FRONTEND, BASE_DIR):
        if (d/f).exists(): return send_from_directory(str(d), f)
    return "Not found", 404

# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/search-stocks", methods=["GET"])
def search_stocks():
    """Live stock search — returns matching instruments."""
    q = request.args.get("q", "").strip()
    if len(q) < 1:
        return jsonify([])
    return jsonify(_search_instruments(q))

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","keras":KERAS_OK,
                    "models":_discover(),"loaded":list(_loaded.keys())})

@app.route("/api/models")
def list_models():
    out = []
    for name in _discover():
        try:
            m = _load(name); ts,nf = _sig(m)
            out.append({"name":name,"timesteps":ts,"features":nf,"params":int(m.count_params())})
        except Exception as e:
            out.append({"name":name,"error":str(e).split("\n")[0]})
    return jsonify(out)


@app.route("/api/market-overview", methods=["POST"])
def market_overview():
    """Live market overview — indices + top stocks."""
    try:
        d     = request.json or {}
        token = d.get("token") or DEFAULT_TOKEN
        if not token:
            return jsonify({"error":"Upstox token required","data":[]}), 400
        data = _market_overview(token)
        return jsonify({"data": data, "timestamp": datetime.now().strftime("%H:%M:%S")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e),"data":[]}), 400


@app.route("/api/quote", methods=["POST"])
def quote():
    """Live LTP for a single ticker."""
    try:
        d      = request.json or {}
        ticker = d.get("ticker","RELIANCE")
        token  = d.get("token") or DEFAULT_TOKEN
        ikey   = _resolve_instrument_key(ticker, d.get("instrument_key"))
        if not ikey:
            return jsonify({"error":f"Ticker {ticker} not found in instruments"}), 400
        q = _upstox_quote(ikey, token)
        if q:
            return jsonify(q)
        return jsonify({"error":"Quote unavailable"}), 400
    except Exception as e:
        return jsonify({"error":str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        d          = request.json or {}
        ticker     = d.get("ticker","RELIANCE.NS")
        model_name = d.get("model") or _best()
        period     = d.get("period","6mo")
        interval   = d.get("interval","1d")
        threshold  = float(d.get("threshold",0.20))
        token      = d.get("token") or d.get("upstox_token") or DEFAULT_TOKEN
        inst_key   = d.get("instrument_key") or None

        model      = _load(model_name)
        ts, nf     = _sig(model)
        df, source = _fetch(ticker, period, interval, token=token, instrument_key=inst_key)
        seq, eng   = _build_seq(df, ts, nf)
        prob       = float(np.clip(model.predict(seq,verbose=0).ravel()[0],0,1))
        band       = _risk_band(prob, threshold)

        # Feature values snapshot for UI display
        feats_list = _pick_features(nf)
        try:
            feat_snap = eng[feats_list].dropna()
            feature_values = {k: round(float(v), 6) for k, v in feat_snap.tail(1).iloc[0].to_dict().items()} if len(feat_snap) > 0 else {}
        except Exception:
            feature_values = {}

        # OHLC for chart — use the actual interval requested so time labels are correct
        try:
            chart_df = _flatten(df).tail(2600)
        except Exception:
            chart_df = _flatten(df).tail(2600)
        def _fmt_date(idx):
            """Always return a UTC ISO string ending in Z for unambiguous parsing."""
            try:
                ts = pd.Timestamp(idx)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert('UTC')
                return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                return str(idx)[:19] + "Z"
        ohlc = [
            {"date":  _fmt_date(idx),
             "open":  round(float(r["Open"]),2),
             "high":  round(float(r["High"]),2),
             "low":   round(float(r["Low"]),2),
             "close": round(float(r["Close"]),2),
             "volume":int(r.get("Volume",0))}
            for idx, r in chart_df.iterrows()
        ]

        # Try to get live LTP from Upstox quote
        live_price = None
        if token:
            ikey = _resolve_instrument_key(ticker, inst_key)
            if ikey:
                q = _upstox_quote(ikey, token)
                if q: live_price = q.get("ltp")

        latest_close = live_price or round(float(df["Close"].iloc[-1]),2)

        return jsonify({
            "ticker":         ticker,
            "model":          model_name,
            "probability":    round(prob,6),
            "risk_pct":       round(prob*100,2),
            "band":           band,
            "threshold":      threshold,
            "timesteps":      ts,
            "features":       nf,
            "ohlc":           ohlc,
            "latest_close":   latest_close,
            "latest_date":    str(df.index[-1])[:10],
            "source":         source,
            "live_price":     live_price,
            "feature_values": feature_values,
            "interval":       interval,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}), 400


@app.route("/api/portfolio", methods=["POST"])
def portfolio():
    try:
        d          = request.json or {}
        tickers    = d.get("tickers",["RELIANCE","TCS","HDFCBANK"])
        model_name = d.get("model") or _best()
        period     = d.get("period","6mo")
        interval   = d.get("interval","1d")
        threshold  = float(d.get("threshold",0.20))
        token      = d.get("token") or d.get("upstox_token") or DEFAULT_TOKEN

        model  = _load(model_name); ts,nf = _sig(model)
        results = []
        inst_keys = d.get("instrument_keys", {})
        for t in tickers:
            try:
                df,source = _fetch(t,period,interval,token=token,instrument_key=inst_keys.get(t))
                seq,_     = _build_seq(df,ts,nf)
                prob      = float(np.clip(model.predict(seq,verbose=0).ravel()[0],0,1))
                results.append({"ticker":t,"probability":round(prob,6),
                                 "risk_pct":round(prob*100,2),"band":_risk_band(prob,threshold),
                                 "latest_close":round(float(df["Close"].iloc[-1]),2),"source":source})
            except Exception as e:
                results.append({"ticker":t,"error":str(e)})
        results.sort(key=lambda x:x.get("probability",0),reverse=True)
        return jsonify({"results":results,"model":model_name,"threshold":threshold})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}), 400


@app.route("/api/rolling-risk", methods=["POST"])
def rolling_risk():
    """Compute rolling crash probability over historical data."""
    try:
        d          = request.json or {}
        ticker     = d.get("ticker", "RELIANCE")
        period     = d.get("period", "6mo")
        interval   = d.get("interval", "1d")
        token      = d.get("token") or DEFAULT_TOKEN
        inst_key   = d.get("instrument_key") or None
        threshold  = float(d.get("threshold", 0.20))
        model_name = d.get("model") or _best()
        model      = _load(model_name)
        ts, nf     = _sig(model)
        df, source = _fetch(ticker, period, interval, token=token, instrument_key=inst_key)
        eng        = _engineer(df)
        feats      = _pick_features(nf)
        ff         = eng[feats].replace([np.inf,-np.inf], np.nan).bfill().ffill().fillna(0)
        if len(ff) < ts + 1:
            return jsonify({"error": f"Need {ts+1} rows, got {len(ff)}"}), 400
        sc      = StandardScaler()
        ff_arr  = ff.to_numpy(np.float32)
        sc.fit(ff_arr)
        ff_s    = sc.transform(ff_arr)
        total   = len(ff) - ts
        step    = max(1, total // 200)
        indices = list(range(ts, len(ff), step))
        if indices[-1] < len(ff) - 1:
            indices.append(len(ff) - 1)
        batch   = np.array([ff_s[i-ts:i] for i in indices], dtype=np.float32)
        preds   = model.predict(batch, verbose=0, batch_size=32).ravel()
        is_intraday = interval in ("1m","5m","15m","30m","1h")
        results = []
        for idx_i, prob in zip(indices, preds):
            results.append({
                "time":        str(ff.index[idx_i])[:19],
                "probability": round(float(np.clip(prob, 0, 1)), 4),
                "close":       round(float(df["Close"].iloc[idx_i]), 2),
            })
        return jsonify({"data": results, "ticker": ticker, "source": source,
                        "threshold": threshold, "is_intraday": is_intraday})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file uploaded"}), 400
        f          = request.files["file"]
        model_name = request.form.get("model") or _best()
        threshold  = float(request.form.get("threshold",0.20))
        model      = _load(model_name); ts,nf = _sig(model)
        df         = _norm(pd.read_csv(io.BytesIO(f.read())))
        seq,eng    = _build_seq(df,ts,nf)
        prob       = float(np.clip(model.predict(seq,verbose=0).ravel()[0],0,1))
        feats      = _pick_features(nf)
        snap       = {k:round(float(v),6) for k,v in eng[feats].dropna().tail(1).iloc[0].to_dict().items()}
        return jsonify({"probability":round(prob,6),"risk_pct":round(prob*100,2),
                         "band":_risk_band(prob,threshold),"threshold":threshold,
                         "model":model_name,"rows_loaded":len(df),"features":snap})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}), 400


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*62)
    print("  ⚡ FlashGuard v4 — Upstox Real-Time Server")
    print("="*62)
    found = _discover()
    print(f"  Models   : {found}")
    print(f"  Frontend : {FRONTEND}")
    print(f"  URL      : http://localhost:5000")
    print(f"  Dashboard: http://localhost:5000/dashboard.html")
    print("="*62)
    for name in found:
        try:   _load(name); print(f"  ✓ {name}")
        except Exception as e: print(f"  ✗ {name}: {e}")
    print()
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
