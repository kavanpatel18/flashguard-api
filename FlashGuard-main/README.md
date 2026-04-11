<div align="center">
  <img src="frontend/Logo.png" alt="FlashGuard Logo" width="180"/>

  <h1>⚡ FlashGuard v4</h1>
  <h3>Real-Time Flash Crash Predictor for Indian Equity Markets</h3>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white"/>
    <img src="https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white"/>
    <img src="https://img.shields.io/badge/Flask-3.0+-green?logo=flask&logoColor=white"/>
    <img src="https://img.shields.io/badge/Upstox-v2_API-7B2FBE?logo=data:image/svg+xml;base64,&logoColor=white"/>
    <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen"/>
  </p>
</div>

---

## 📖 About

FlashGuard is a production-grade **Flash Crash Prediction Engine** for Indian stock markets. It uses a deep learning pipeline — specifically a **GRU (Gated Recurrent Unit) architecture with a custom Attention mechanism** — to score the probability of a sudden liquidity-driven price collapse for any NSE/BSE instrument.

The system pulls **real-time OHLCV candles** from the Upstox v2 API, engineers 10 proprietary risk features, runs them through the trained model, and outputs an actionable risk band — all in under a second.

---

## 🏗️ App Architecture

```
FlashGuard-main/
├── api_server.py                       # Core Flask API (all routes, model loading, data fetch)
├── improved_minute_model.keras         # Primary model  — 30 timesteps × 10 features
├── improved_flash_crash_model.keras    # Secondary / fallback model
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Hugging Face Spaces deployment
├── Procfile                            # Heroku / Render process definition
├── render.yaml                         # Render service config
└── frontend/
    ├── index.html                      # Landing page / splash screen
    ├── dashboard.html                  # Active trading terminal dashboard
    ├── style.css                       # Dark-mode UI design system
    ├── app.js                          # Frontend logic, Chart.js, API calls
    └── Logo.png                        # FlashGuard branding
```

---

## ⚙️ Local Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python api_server.py
```

| Page        | URL                                         |
|-------------|---------------------------------------------|
| Landing     | `http://localhost:5000/`                    |
| Dashboard   | `http://localhost:5000/dashboard.html`      |

---

## 🔑 Upstox API Token

For live market data, you need an Upstox OAuth 2.0 access token:
1. Register at [developer.upstox.com](https://developer.upstox.com/)
2. Create an app and authenticate via the OAuth flow
3. Paste the generated token into the **Token** field in the FlashGuard dashboard

> 🟡 Without a token, FlashGuard automatically falls back to **yFinance** (delayed) and then **synthetic demo data** — so it always works.

---

## 📡 API Reference

| Method | Endpoint               | Description                              |
|--------|------------------------|------------------------------------------|
| `GET`  | `/api/health`          | Server status, loaded models             |
| `GET`  | `/api/models`          | All available prediction models          |
| `GET`  | `/api/search-stocks`   | Live instrument search (5000+ NSE/BSE)   |
| `POST` | `/api/predict`         | Flash crash risk for a single stock      |
| `POST` | `/api/portfolio`       | Batch risk scan across a portfolio       |
| `POST` | `/api/rolling-risk`    | Historical rolling risk probability      |
| `POST` | `/api/quote`           | Live LTP quote via Upstox                |
| `POST` | `/api/market-overview` | Live overview of indices & top stocks    |
| `POST` | `/api/upload`          | Predict from a custom uploaded CSV file  |

### Example — Predict crash risk for RELIANCE
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE", "period": "6mo", "interval": "1d"}'
```

**Response:**
```json
{
  "ticker": "RELIANCE",
  "probability": 0.073,
  "risk_pct": 7.3,
  "band": "STABLE",
  "threshold": 0.2,
  "source": "upstox",
  "live_price": 1248.50
}
```

---

## 🧠 Model Details

| Property       | Value                                        |
|----------------|----------------------------------------------|
| Architecture   | GRU + Custom Attention Layer (Keras)         |
| Input Shape    | `(1, 30, 10)` — 30 timesteps × 10 features  |
| Output         | Sigmoid probability `[0, 1]`                 |
| Training Data  | 90K+ rows of NSE minute-level OHLCV data     |

### 10 Engineered Input Features
| Feature              | Description                           |
|----------------------|---------------------------------------|
| `return`             | Simple period-over-period return      |
| `log_return`         | Log-scale return (noise-stabilised)   |
| `volatility_5`       | Rolling 5-period std. deviation       |
| `volatility_10`      | Rolling 10-period std. deviation      |
| `volatility_20`      | Rolling 20-period std. deviation      |
| `momentum_5`         | 5-period price momentum               |
| `momentum_10`        | 10-period price momentum              |
| `high_low_spread`    | Intrabar range relative to close      |
| `open_close_return`  | Bar-open to bar-close return          |
| `price_acceleration` | Second derivative of returns          |

### Risk Band Classification
| Band        | Probability Threshold | Meaning                        |
|-------------|-----------------------|--------------------------------|
| 🟢 STABLE   | `< 13%`               | Low crash risk                 |
| 🟡 ELEVATED | `13% – 20%`           | Moderate risk, monitor closely |
| 🔴 HIGH RISK| `≥ 20%`               | Significant crash signal       |

---

## 🚀 Deployment

### Render (Gunicorn)
```bash
# Deployed automatically using render.yaml
gunicorn api_server:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
```

### Hugging Face Spaces (Docker)
```bash
# Built from Dockerfile on every push to main
docker build -t flashguard .
docker run -p 5000:7860 flashguard
```

---

## 📄 License
MIT — Free to use, modify, and distribute with attribution.
