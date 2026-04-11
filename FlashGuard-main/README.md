---
title: FlashGuard
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# ⚡ FlashGuard — Stock Market Flash Crash Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-green?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

FlashGuard is a real-time stock market flash crash prediction system powered by GRU-based deep learning models with attention mechanisms. It monitors Indian equity markets (NSE/BSE) and provides crash risk assessments using live market data from Upstox API.

---

## 🚀 Features

- **Real-Time Prediction** — Live crash risk scoring using trained Keras models
- **Upstox API Integration** — Primary data source for real-time NSE/BSE market data
- **yFinance Fallback** — Automatic fallback to Yahoo Finance when Upstox is unavailable
- **Dynamic Stock Search** — Search across 5000+ NSE/BSE instruments in real-time
- **Portfolio Scanner** — Scan multiple stocks simultaneously for crash risk
- **Interactive Dashboard** — Professional dark-mode UI with OHLC candlestick charts
- **CSV Upload** — Upload custom OHLC data for offline crash risk analysis
- **Market Overview** — Live indices and top stock prices at a glance

## 📐 Architecture

```
FlashGuard/
├── api_server.py                       # Flask API server (backend)
├── improved_minute_model.keras         # GRU + Attention model (30×10)
├── improved_flash_crash_model.keras    # GRU + Attention model (alternate)
├── frontend/
│   ├── index.html                      # Landing page
│   ├── dashboard.html                  # Dashboard with charts & predictions
│   ├── style.css                       # Dark-mode UI styles
│   ├── app.js                          # Frontend logic & API calls
│   └── Logo.png                        # FlashGuard brand logo
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Hugging Face Spaces deployment
├── Procfile                            # Heroku/Render start command
├── render.yaml                         # Render deployment config
└── README.md
```

## 🛠️ Tech Stack

| Layer      | Technology                                      |
| ---------- | ----------------------------------------------- |
| Backend    | Python 3.11, Flask, Flask-CORS                  |
| ML Models  | TensorFlow / Keras (GRU + Custom Attention)     |
| Data       | Upstox v2 API (primary), yFinance (fallback)    |
| Frontend   | Vanilla HTML/CSS/JavaScript, Chart.js           |
| Deployment | Hugging Face Spaces (Docker), Render (Gunicorn) |

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Godar72/FlashGuard.git
cd FlashGuard

# Install dependencies
pip install -r requirements.txt

# Run the server
python api_server.py
```

The server starts at **http://localhost:5000**

- Landing Page: `http://localhost:5000/`
- Dashboard: `http://localhost:5000/dashboard.html`

## 🤗 Deploy on Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Connect your GitHub repo or push directly to the Space
4. The app auto-deploys and is available at `https://huggingface.co/spaces/YOUR_USERNAME/FlashGuard`

## 🔑 Upstox API Token

FlashGuard uses the Upstox v2 REST API for real-time market data. To get live data:

1. Create an account at [Upstox Developer](https://developer.upstox.com/)
2. Generate an access token via OAuth 2.0
3. Paste the token in the dashboard's token input field

> Without a token, the system falls back to yFinance (delayed data) or demo data.

## 📡 API Endpoints

| Method | Endpoint               | Description                     |
| ------ | ---------------------- | ------------------------------- |
| GET    | `/api/health`          | Server health & model status    |
| GET    | `/api/models`          | List available prediction models|
| GET    | `/api/search-stocks`   | Search NSE/BSE instruments      |
| POST   | `/api/predict`         | Single stock crash prediction   |
| POST   | `/api/portfolio`       | Multi-stock portfolio scan      |
| POST   | `/api/quote`           | Live price quote                |
| POST   | `/api/market-overview` | Major indices & stocks overview |
| POST   | `/api/upload`          | Upload CSV for prediction       |

## 🧠 Model Details

- **Architecture**: GRU (Gated Recurrent Unit) + Custom Attention Layer
- **Input Shape**: 30 timesteps × 10 features
- **Features**: Returns, log returns, volatility (5/10/20), momentum, spread metrics
- **Output**: Flash crash probability (0–1) with risk bands:
  - 🟢 **STABLE** — Low crash risk
  - 🟡 **ELEVATED** — Moderate risk, monitor closely
  - 🔴 **HIGH RISK** — Significant crash probability detected

## 🌐 Deployment (Render)

This project is also configured for deployment on [Render](https://render.com):

```yaml
# render.yaml
services:
  - type: web
    name: flashguard
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn api_server:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
```

## 📄 License

This project is part of a PBL (Project Based Learning) initiative.

---

**Built with ❤️ by Godar72**
