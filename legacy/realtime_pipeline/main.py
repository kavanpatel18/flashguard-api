import asyncio
import logging
import yaml
import time
from ingestion import KafkaStreamIngestion, HftOrderBook
from processing import DeepLOBFeatureEngine, SequenceBuffer
from inference import FlashCrashInference, AlertGateway

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SystemRunner")

async def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 1. Init Streaming (Kafka/WS Pattern)
    raw_topic_q = asyncio.Queue()
    ingress = KafkaStreamIngestion(raw_topic_q, config['kafka']['bootstrap_servers'], config['kafka']['raw_topic'])
    
    # 2. Init Event-Driven L2 Book (hftbacktest Pattern)
    order_book = HftOrderBook(config['market']['symbol'])
    
    # 3. Init ML-HFT & DeepLOB Processing Modules
    feature_engine = DeepLOBFeatureEngine(levels=config['model']['levels'])
    buffer = SequenceBuffer(seq_len=config['model']['sequence_length'], num_features=config['model']['num_features'])
    
    # 4. Init Inference Pipeline
    inference = FlashCrashInference(
        threshold=config['model']['threshold'], 
        num_features=config['model']['num_features'], 
        seq_len=config['model']['sequence_length']
    )
    gateway = AlertGateway(threshold=config['model']['threshold'])

    await ingress.start()

    logger.info("Initializing Real-Time Stream Processing Topology...")
    
    snapshot_interval = config['app']['snapshot_interval_ms'] / 1000.0
    last_snapshot_ts = time.time()

    try:
        while True:
            # Event Loop: Process all pending tick/order events (HFT Pattern)
            while not raw_topic_q.empty():
                msg = await raw_topic_q.get()
                if msg['type'] == 'snapshot':
                    order_book.apply_snapshot(msg['bids'], msg['asks'], msg['ts'])
                elif msg['type'] == 'tick':
                    order_book.apply_tick(msg['side'], msg['price'], msg['qty'], msg['ts'])
                    
            # Temporal Sampling: Export Features (DeepLOB Pattern)
            current_time = time.time()
            if (current_time - last_snapshot_ts) >= snapshot_interval:
                last_snapshot_ts = current_time
                
                # Extract Top N Book State
                top_lob = order_book.get_top_n(config['model']['levels'])
                features = feature_engine.compute(top_lob)
                
                buffer.push(features)
                
                if buffer.is_warmed_up():
                    tensor = buffer.get_batch()
                    risk_score = inference.evaluate_risk(tensor)
                    gateway.send(risk_score)

            # Prevent CPU starvation
            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        logger.info("Shutdown signal received.")
    finally:
        ingress.running = False

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System Halted.")
