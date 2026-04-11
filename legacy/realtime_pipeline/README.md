# Real-Time Flash Crash Predictor (Unified ML-HFT Architecture)

This project integrates state-of-the-art patterns from several leading quantitative repositories:

- **Streaming / Ingestion:** Incorporates decoupled Kafka Producer/Consumer abstractions suitable for high-frequency tick data.
- **`hftbacktest` (Order Book):** Implements an event-driven continuous L2 limit order book matching its standard `apply_snapshot` and `apply_tick` paradigms to handle atomic BBO updates effectively.
- **`DeepLOB` (Data Structure):** Maps the LOB depth into a dense `(Batch, Sequence, Features)` tensor structure (vectorizing 10 layers of price/volume data).
- **`ML-HFT` (Signals):** Augments raw tensor levels with engineered microstructure features including Microprice, Depth Ratios, and Order Flow Imbalance.
- **Conv-Transformer (Model):** Employs spatial 1D Convolutions (`DeepLOB` pattern) coupled with `TransformerEncoder` blocks for deep temporal risk inference.

## Folder Structure
```
.
├── config.yaml          # Hyperparameters and application settings
├── ingestion.py         # Mock Kafka Stream & hftbacktest L2 Reconstruction
├── processing.py        # Microstructure feature math and DeepLOB tensor logic
├── model.py             # Conv-Transformer PyTorch architecture
├── inference.py         # Low-latency inference wrapper & risk alerting 
├── main.py              # Async pipeline orchestration 
├── requirements.txt     # Dependencies
└── README.md            # You are here
```

## Run Instructions

1. `pip install -r requirements.txt`
2. Configure settings inside `config.yaml` 
3. Execute pipeline
```bash
python main.py
```
