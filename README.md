<div align="center">
  <img src="FlashGuard-main/frontend/Logo.png" alt="FlashGuard Logo" width="200"/>
  <h1>⚡ FlashGuard v4</h1>
  <p><strong>Real-Time Stock Market Flash Crash Predictor using Attention-based Deep Learning</strong></p>

  <p>
    <a href="https://github.com/kavanpatel18/flashguard-api/stargazers"><img src="https://img.shields.io/github/stars/kavanpatel18/flashguard-api?style=for-the-badge&color=yellow" alt="Stars Badge"/></a>
    <a href="https://github.com/kavanpatel18/flashguard-api/network/members"><img src="https://img.shields.io/github/forks/kavanpatel18/flashguard-api?style=for-the-badge&color=blue" alt="Forks Badge"/></a>
    <a href="https://github.com/kavanpatel18/flashguard-api/issues"><img src="https://img.shields.io/github/issues/kavanpatel18/flashguard-api?style=for-the-badge&color=red" alt="Issues Badge"/></a>
    <a href="https://github.com/kavanpatel18/flashguard-api/blob/main/LICENSE"><img src="https://img.shields.io/github/license/kavanpatel18/flashguard-api?style=for-the-badge&color=blueviolet" alt="License Badge"/></a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python Badge"/>
    <img src="https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white" alt="TensorFlow Badge"/>
    <img src="https://img.shields.io/badge/Flask-3.0+-green?logo=flask&logoColor=white" alt="Flask Badge"/>
    <img src="https://img.shields.io/badge/Upstox-v2_API-purple?logo=upstox&logoColor=white" alt="Upstox Badge"/>
    <a href="https://flashguard-api.onrender.com"><img src="https://img.shields.io/badge/Live_Demo-🟢_Online-brightgreen?style=flat-square" alt="Live Demo"/></a>
  </p>

  <h3>🌐 <a href="https://flashguard-api.onrender.com">Live Demo → flashguard-api.onrender.com</a></h3>
</div>

---

## 📖 Overview

**FlashGuard** is an advanced, real-time stock market crash prediction system. Designed to identify the early warning signs of liquidity sinkholes and algorithmic "flash crashes", it monitors Indian equity markets (NSE/BSE) using live data streams. 

Under the hood, FlashGuard leverages a custom **Gated Recurrent Unit (GRU)** layered with a mathematical **Attention Mechanism**, processing high-frequency OHLCV (Open, High, Low, Close, Volume) data and sophisticated technical derivatives (volatility, momentum, spreads) to output live probabilistic risk scores.

### ✨ Key Features
- 🔴 **Real-Time Prediction:** Live crash probability scoring utilizing custom Keras algorithms.
- 📡 **Upstox API Integration:** Tier-1 integration for high-speed live market quoting, with a steady `yFinance` fallback.
- 🔍 **Dynamic Portfolio Scanner:** Search through over 5000+ NSE/BSE instruments instantly.
- 📈 **Interactive Charting Engine:** Immersive, dark-mode terminal UI with interactive candlestick charts showing live market changes.
- 🧠 **Explainable AI (XAI) Snapshots:** Transparent display of the exact 10 core features (like log returns, rolling volatility, etc.) used to determine current risk bands. 

---

## 📁 Project Architecture 

This repository conforms to a professional Machine Learning project standard, explicitly separating the core application from the data engineering pipelines.

```bash
flashguard-api/
├── FlashGuard-main/        # 🚀 THE APPLICATION: Flask backend + frontend UI
├── data/                   # 📊 DATASETS: Raw minute-level data & historical files
├── docs/                   # 📄 RESEARCH: Academic papers & presentation decks
├── legacy/                 # 🗑️ ARCHIVE: Sandbox, experimental clones, and old tests
├── models/                 # 🧠 MODELS: Pre-trained .keras weight artifacts
├── notebooks/              # 📓 NOTEBOOKS: Jupyter EDA, feature engineering, and analytics
├── scripts/                # ⚙️ SCRIPTS: Model training and standalone anomaly engines
├── .gitignore              # GitHub ignores (venv, cache, raw data)
├── LICENSE                 # MIT License terms
└── README.md               # You are here!
```

---

## 🛠️ Getting Started (Running the App)

Running the production FlashGuard server locally is simple.

### 1. Prerequisites
Ensure you have **Python 3.11+** installed on your system. 

### 2. Installation
Navigate directly into the production application folder:
```bash
git clone https://github.com/kavanpatel18/flashguard-api.git
cd flashguard-api/FlashGuard-main

# Install all required Python packages
pip install -r requirements.txt
```

### 3. Launching the Market Server
Start the real-time API server:
```bash
python api_server.py
```
*   **Landing Page:** Access `http://localhost:5000/` in your browser.
*   **Active Dashboard:** Access `http://localhost:5000/dashboard.html` to begin predicting.

---

## 🔑 Upstox API Token (Live Data)

To experience true real-time parsing, FlashGuard connects to Upstox.
1. Create an account at the [Upstox Developer Platform](https://developer.upstox.com/).
2. Setup a new App and generate an **OAuth 2.0 Access Token**.
3. Paste the token into the FlashGuard Dashboard input field.
*(Note: If no token is provided, FlashGuard safely falls back to delayed Yahoo Finance quotes or synthetic demonstration data).*

---

## 🧠 Machine Learning & Retraining

For data scientists interested in the underlying algorithms:
- **Model Architecture:** The core model is an MSA-GRU array (Multi-Scale Attention GRU) built in TensorFlow. It looks across 30 distinct time-steps featuring 10 technical indicators.
- **Dataset Generation:** Explore `/notebooks/nifty50_flash_crash_dataset_builder.ipynb` to see how the raw minute-level datasets (found in `/data/`) are filtered and engineered.
- **Retraining:** The standalone training files are completely isolated in the `/scripts/` folder (e.g., `train_minute.py`). Any heavily modified Keras files are outputted directly to the `/models/` directory for safe versioning.

---

## 🤝 Contributing

We welcome contributions to optimize the anomaly detection parameters or expand exchange support. 
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License & Acknowledgements

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Built natively for highly-efficient stock processing and flash crash detection research.

<div align="center">
  <b>Built with ❤️ by Kavan Patel</b>
</div>
