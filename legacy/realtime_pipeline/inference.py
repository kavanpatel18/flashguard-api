import torch
import logging
import numpy as np
from model import DeepLOBConvTransformer

logger = logging.getLogger(__name__)

class FlashCrashInference:
    """Predicts crash probability from the continuous DeepLOB LOB sequence."""
    def __init__(self, threshold: float, num_features: int, seq_len: int):
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepLOBConvTransformer(num_features=num_features, seq_len=seq_len).to(self.device)
        self.model.eval()
        # Compile model for ultra-low latency in PyTorch 2.0
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        logger.info(f"Model Engine initialized. Alert Threshold: {self.threshold}")

    def evaluate_risk(self, tensor: np.ndarray) -> float:
        with torch.inference_mode():
            x = torch.tensor(tensor, dtype=torch.float32, device=self.device)
            prob = self.model(x).item()
            return prob

class AlertGateway:
    """Simulates pushing alerts to Kafka or external services."""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def send(self, prob: float):
        if prob >= self.threshold:
            logger.critical(f"🚨 [FLASH CRASH DETECTED] Prob: {prob:.4f} > {self.threshold}")
        else:
            logger.debug(f"Risk OK: {prob:.4f}")
