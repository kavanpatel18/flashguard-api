import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from model import DeepLOBConvTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML_Trainer")

def generate_synthetic_data(num_samples=1000, seq_len=100, num_features=45):
    """
    Generates dummy order book tensors mapping to the DeepLOB pipeline architecture.
    Simulates flash crash events for supervised training.
    """
    X = np.random.randn(num_samples, seq_len, num_features).astype(np.float32)
    # 5% chance of being a flash crash label
    y = np.random.choice([0.0, 1.0], size=(num_samples, 1), p=[0.95, 0.05]).astype(np.float32)
    
    # Inject synthetic pattern: If it's a crash, drop the micro-price (feature index 40) rapidly in last 10 ticks
    for i in range(num_samples):
        if y[i, 0] == 1.0:
            X[i, -10:, 40] -= np.linspace(0, 5, 10)
            
    return torch.tensor(X), torch.tensor(y)

def train_model():
    os.makedirs('weights', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLOBConvTransformer(num_features=45, seq_len=100).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    logger.info("Generating synthetic tensor sequences for ML-HFT architecture...")
    X_train, y_train = generate_synthetic_data(2000)
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    epochs = 10
    batch_size = 32
    
    logger.info(f"Starting Training on {device}...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_train):.4f}")
        
    torch.save(model.state_dict(), 'weights/conv_transformer.pth')
    logger.info("Training Complete. Weights saved to weights/conv_transformer.pth")

if __name__ == "__main__":
    train_model()
