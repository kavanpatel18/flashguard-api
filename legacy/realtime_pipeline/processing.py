import numpy as np
import collections
from typing import Dict, Any, List

class DeepLOBFeatureEngine:
    """
    Combines DeepLOB raw order book state arrays with ML-HFT engineered signals.
    """
    def __init__(self, levels: int = 10):
        self.levels = levels
        self.prev_microprice = None

    def compute(self, lob_state: Dict[str, Any]) -> List[float]:
        bids = lob_state.get('bids', [])
        asks = lob_state.get('asks', [])
        
        # DeepLOB Tensor Structure: [P_ask, V_ask, P_bid, V_bid] repeated for N levels
        # Shape: (4 * levels) = 40 properties
        tensor_features = []
        for i in range(self.levels):
            a_p, a_v = asks[i] if i < len(asks) else (0.0, 0.0)
            b_p, b_v = bids[i] if i < len(bids) else (0.0, 0.0)
            tensor_features.extend([a_p, a_v, b_p, b_v])
            
        # ML-HFT Features
        # 1. Micro-Price (Volume Weighted Mid)
        if len(bids) > 0 and len(asks) > 0:
            p_a1, v_a1 = asks[0]
            p_b1, v_b1 = bids[0]
            micro_price = (p_a1 * v_b1 + p_b1 * v_a1) / (v_a1 + v_b1) if (v_a1 + v_b1) > 0 else (p_a1 + p_b1) / 2
        else:
            micro_price = 0.0
            
        # 2. Imbalance (L1)
        imbalance = (v_b1 - v_a1) / (v_b1 + v_a1) if len(bids) > 0 and (v_a1 + v_b1) > 0 else 0.0
        
        # 3. Accumulated Depth Ratio
        depth_b = sum([v for p, v in bids])
        depth_a = sum([v for p, v in asks])
        depth_ratio = depth_b / depth_a if depth_a > 0 else 1.0
        
        # 4. Spread
        spread = p_a1 - p_b1 if len(bids) > 0 else 0.0
        
        # 5. Micro-Price Return
        ret = np.log(micro_price / self.prev_microprice) if self.prev_microprice and self.prev_microprice > 0 else 0.0
        self.prev_microprice = micro_price

        # Total Features: 40 + 5 = 45
        return tensor_features + [micro_price, imbalance, depth_ratio, spread, ret]

class SequenceBuffer:
    """Sliding window for Conv-Transformer (DeepLOB temporal tracking)"""
    def __init__(self, seq_len: int, num_features: int):
        self.seq_len = seq_len
        self.buffer = collections.deque(maxlen=seq_len)
        self.num_features = num_features
        # Pre-fill with zeros for immediate readiness, tracking valid steps
        for _ in range(seq_len):
             self.buffer.append([0.0] * num_features)
        self.valid_steps = 0

    def push(self, vector: List[float]):
        self.buffer.append(vector)
        self.valid_steps = min(self.valid_steps + 1, self.seq_len)

    def is_warmed_up(self) -> bool:
        return self.valid_steps >= self.seq_len

    def get_batch(self) -> np.ndarray:
        return np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)
