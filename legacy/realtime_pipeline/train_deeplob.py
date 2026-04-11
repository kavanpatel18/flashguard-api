import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from model import DeepLOBConvTransformer
from dataset import get_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepLOB_Trainer")

def train_model():
    os.makedirs('weights', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Intialize ConvTransformer matching exactly the 45 features and sliding window
    model = DeepLOBConvTransformer(num_features=45, seq_len=60).to(device)
    
    # Exact DeepLOB loss and Adam optimization
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    logger.info("Initializing Real Binance Data Loader Pipeline...")
    # Fetch snapshots to construct a robust dataset using pure deep learning structures
    train_loader, val_loader = get_dataloaders(num_snapshots=500, seq_len=60, batch_size=32)
    
    epochs = 10
    logger.info(f"Starting Genuine Training on {device} using DeepLOB methodology...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).float().unsqueeze(1)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), 'weights/deeplob_binance.pth')
    logger.info("Training Complete. Model saved to weights/deeplob_binance.pth")

if __name__ == "__main__":
    train_model()
