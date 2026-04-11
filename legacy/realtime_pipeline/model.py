import torch
import torch.nn as nn

class DeepLOBConvTransformer(nn.Module):
    """
    Merges DeepLOB's spatial CNN feature extraction on raw tensor [Pa, Va, Pb, Vb]
    with a Transformer Encoder for long-range temporal dependencies.
    """
    def __init__(self, num_features=45, seq_len=100, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # Inception-style Spatial Convolution (1x1 and 3x1 mapping for DeepLOB structures)
        # Assuming Num Features is handled as channel dimension
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        
        # Transformer Temporal Encode
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Risk Head
        self.fc = nn.Sequential(
            nn.Linear(d_model * seq_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        x = x.transpose(1, 2)  # (Batch, Features, Seq_Len) for Conv1d
        
        # Spatial-Temporal Conv block (DeepLOB style feature reduction)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Attention over temporal sequence
        x = x.transpose(1, 2)  # (Batch, Seq_Len, d_model)
        x = self.transformer(x)
        
        # Flatten and predict crash probability
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out
