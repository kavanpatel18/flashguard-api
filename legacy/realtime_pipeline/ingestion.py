import asyncio
import logging
import random
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HftOrderBook:
    """
    Event-driven L2 Order Book adapted from hftbacktest patterns.
    Handles 'snapshot' (full replacement) and 'tick' (incremental update) events.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update_ts = 0

    def apply_snapshot(self, bids: list, asks: list, ts: int):
        self.bids = {price: qty for price, qty in bids}
        self.asks = {price: qty for price, qty in asks}
        self.last_update_ts = ts

    def apply_tick(self, side: str, price: float, qty: float, ts: int):
        target_book = self.bids if side == 'bid' else self.asks
        if qty <= 0:
            target_book.pop(price, None)
        else:
            target_book[price] = qty
        self.last_update_ts = ts

    def get_top_n(self, n: int = 10) -> Dict[str, list]:
        """Returns sorted top N levels for DeepLOB tensor structure."""
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:n]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return {
            "bids": sorted_bids,
            "asks": sorted_asks,
            "ts": self.last_update_ts
        }

class KafkaStreamIngestion:
    """
    Abstracts WebSocket to Kafka streaming patterns. 
    In production, this bridges aiokafka Producers and Consumers.
    Mocks the behavior for standalone runnable demonstration.
    """
    def __init__(self, queue: asyncio.Queue, bootstrap: str, topic: str):
        self.queue = queue
        self.bootstrap = bootstrap
        self.topic = topic
        self.running = False
        self.base_price = 22000.0

    async def start(self):
        self.running = True
        logger.info(f"Connected to Upstox WS -> Kafka Topic: {self.topic}")
        asyncio.create_task(self._mock_kafka_consumer())

    async def _mock_kafka_consumer(self):
        """Mocks reading from Kafka `raw_ticks` partition"""
        while self.running:
            self.base_price += random.uniform(-1, 1)
            
            # Simulate a snapshot followed by ticks
            bids = [[self.base_price - (i * 0.5), random.randint(10, 100)] for i in range(1, 15)]
            asks = [[self.base_price + (i * 0.5), random.randint(10, 100)] for i in range(1, 15)]

            msg = {
                'type': 'snapshot',
                'ts': asyncio.get_event_loop().time(),
                'bids': bids,
                'asks': asks
            }
            await self.queue.put(msg)
            await asyncio.sleep(0.01) # 10ms high-frequency
