# ⚡ FlashGuard API System

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

A dual-repository real-time API Server acting as the risk assessment brain for high-frequency stock trading. FlashGuard actively monitors live ticker data (Bank Nifty, Reliance, etc.) from the Upstox v2 API (or delayed YFinance fallbacks) and streams the data directly through heavily customized deep learning models.

## 🚀 Architecture
- **Inference Server**: Runs the live `msa_gru_best.keras` model, a custom **Multi-Scale Attention GRU** detecting volatile anomalies in live candlestick windows.
- **Real-Time Data Pipeling**: Interfaces natively with REST protocols, pushing data boundaries into Numpy scaling mechanics dynamically.
- **TradingView Interface**: Contains a seamless `app.js` and `dashboard.html` charting interface dynamically fetching real-time predictions via CORS architecture.

## 🛠️ Deployment
```bash
pip install -r requirements.txt
python api_server.py
```
Server runs on `http://localhost:5000`.
