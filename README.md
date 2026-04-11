# ⚡ FlashGuard — Stock Market Flash Crash Predictor (ML Repository)

This repository contains the complete Machine Learning pipeline and source code for **FlashGuard**, a real-time stock market flash crash prediction system powered by GRU-based deep learning models with custom attention mechanisms.

## 📁 Repository Structure

*   **/FlashGuard-main** - The production-ready FlashGuard application (Flask backend + HTML/JS frontend). To run the app, navigate here and follow the enclosed instructions.
*   **/data** - Datasets and raw minute-level OHLC data used for model training. (Note: large CSV files are ignored via `.gitignore`).
*   **/notebooks** - Jupyter Notebooks detailing Exploratory Data Analysis (EDA), dataset merging, and model tuning.
*   **/scripts** - Standalone Python training scripts (`train_minute.py`, anomaly engine components, and custom layer definitions).
*   **/models** - Compiled Keras model artifacts (`.keras` weights) and training results tracking.
*   **/docs** - Project presentations, PDFs, and background research papers.
*   **/legacy** - Deprecated configurations, old experimental iterations, and redundant clones.

## 🚀 Getting Started

If you want to run the FlashGuard real-time prediction dashboard:
```bash
cd FlashGuard-main
pip install -r requirements.txt
python api_server.py
```
View the enclosed `FlashGuard-main/README.md` for detailed instructions on obtaining the Upstox API Token, using the portfolio scanner, and interacting with the system.

## 🧠 Model Training

To retrain the models:
1. Navigate to `/notebooks` to explore the dataset building process.
2. Execute the training pipelines via the files in the `/scripts` directory.

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
