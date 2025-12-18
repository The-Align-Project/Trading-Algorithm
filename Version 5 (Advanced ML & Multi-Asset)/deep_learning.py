"""Deep Learning models for price prediction - LSTM and Transformer architectures."""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from data_structures import DeepLearningPrediction, ModelEnsemblePrediction
from config import PYTORCH_AVAILABLE, TradingConfig

if PYTORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

class PriceDataset(Dataset):
    """Dataset for price sequences"""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMPricePredictor(nn.Module):
    """LSTM model for price prediction"""

    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPricePredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        out = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class TransformerPricePredictor(nn.Module):
    """Transformer model for price prediction"""

    def __init__(self, input_size=10, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerPricePredictor, self).__init__()

        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layers
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoding
        transformer_out = self.transformer_encoder(x)

        # Use the last output
        out = transformer_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DeepLearningPredictor:
    """Manager for deep learning models"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lstm_model = None
        self.transformer_model = None
        self.is_trained = False

        if PYTORCH_AVAILABLE:
            self.logger.info(f"Deep learning models initialized on {self.device}")
        else:
            self.logger.warning("PyTorch not available. Deep learning features disabled.")

    def prepare_sequences(self, data: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        if len(data) < sequence_length + 1:
            return np.array([]), np.array([])

        # Select features for model input
        features = [
            'Close', 'Volume', 'RSI', 'MACD', 'BB_Position',
            'ATR_Norm', 'SMA_Ratio', 'EMA_Ratio', 'Trend_Strength', 'Volume_Ratio'
        ]

        # Handle missing features
        available_features = [f for f in features if f in data.columns]

        if len(available_features) < 3:
            self.logger.warning("Insufficient features for deep learning")
            return np.array([]), np.array([])

        # Normalize features
        feature_data = data[available_features].fillna(method='ffill').fillna(0)
        normalized_data = (feature_data - feature_data.mean()) / (feature_data.std() + 1e-8)

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(normalized_data) - sequence_length):
            seq = normalized_data.iloc[i:i+sequence_length].values
            target = data['Close'].iloc[i+sequence_length] / data['Close'].iloc[i+sequence_length-1] - 1

            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def train_models(self, data: pd.DataFrame, epochs: int = 50):
        """Train both LSTM and Transformer models"""
        if not PYTORCH_AVAILABLE:
            self.logger.warning("PyTorch not available. Cannot train models.")
            return

        self.logger.info("Preparing training data...")
        sequences, targets = self.prepare_sequences(data)

        if len(sequences) == 0:
            self.logger.warning("Not enough data to train models")
            return

        # Split into train/validation
        split_idx = int(len(sequences) * 0.8)
        train_sequences = sequences[:split_idx]
        train_targets = targets[:split_idx]
        val_sequences = sequences[split_idx:]
        val_targets = targets[split_idx:]

        # Create datasets
        train_dataset = PriceDataset(train_sequences, train_targets)
        val_dataset = PriceDataset(val_sequences, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=TradingConfig.DL_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=TradingConfig.DL_BATCH_SIZE)

        # Initialize models
        input_size = train_sequences.shape[2]
        self.lstm_model = LSTMPricePredictor(input_size=input_size).to(self.device)
        self.transformer_model = TransformerPricePredictor(input_size=input_size).to(self.device)

        # Train LSTM
        self.logger.info("Training LSTM model...")
        self._train_model(self.lstm_model, train_loader, val_loader, epochs)

        # Train Transformer
        self.logger.info("Training Transformer model...")
        self._train_model(self.transformer_model, train_loader, val_loader, epochs)

        self.is_trained = True
        self.logger.info("Deep learning models trained successfully!")

    def _train_model(self, model, train_loader, val_loader, epochs):
        """Train a single model"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=TradingConfig.DL_LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0

            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1)
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    def predict(self, data: pd.DataFrame, symbol: str) -> ModelEnsemblePrediction:
        """Make predictions using both models"""
        if not PYTORCH_AVAILABLE or not self.is_trained:
            # Return default prediction if models not available
            return ModelEnsemblePrediction(
                symbol=symbol,
                lstm_prediction=None,
                transformer_prediction=None,
                ml_prediction=0.0,
                ensemble_prediction=0.0,
                confidence=0.0
            )

        # Prepare input sequence
        sequences, _ = self.prepare_sequences(data, TradingConfig.DL_SEQUENCE_LENGTH)

        if len(sequences) == 0:
            return ModelEnsemblePrediction(
                symbol=symbol,
                lstm_prediction=None,
                transformer_prediction=None,
                ml_prediction=0.0,
                ensemble_prediction=0.0,
                confidence=0.0
            )

        # Get last sequence
        last_sequence = torch.FloatTensor(sequences[-1:]).to(self.device)

        # LSTM prediction
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(last_sequence).item()

        lstm_prediction = DeepLearningPrediction(
            symbol=symbol,
            model_type='LSTM',
            prediction=lstm_pred,
            confidence=0.7,
            price_targets={
                'short': data['Close'].iloc[-1] * (1 + lstm_pred),
                'medium': data['Close'].iloc[-1] * (1 + lstm_pred * 1.5),
                'long': data['Close'].iloc[-1] * (1 + lstm_pred * 2)
            },
            volatility_forecast=abs(lstm_pred)
        )

        # Transformer prediction
        self.transformer_model.eval()
        with torch.no_grad():
            transformer_pred = self.transformer_model(last_sequence).item()

        transformer_prediction = DeepLearningPrediction(
            symbol=symbol,
            model_type='Transformer',
            prediction=transformer_pred,
            confidence=0.75,
            price_targets={
                'short': data['Close'].iloc[-1] * (1 + transformer_pred),
                'medium': data['Close'].iloc[-1] * (1 + transformer_pred * 1.5),
                'long': data['Close'].iloc[-1] * (1 + transformer_pred * 2)
            },
            volatility_forecast=abs(transformer_pred)
        )

        # Ensemble prediction (weighted average)
        ensemble_pred = (lstm_pred * 0.5) + (transformer_pred * 0.5)
        confidence = min(0.8, (abs(ensemble_pred) * 10))  # Higher confidence for stronger signals

        return ModelEnsemblePrediction(
            symbol=symbol,
            lstm_prediction=lstm_prediction,
            transformer_prediction=transformer_prediction,
            ml_prediction=ensemble_pred,
            ensemble_prediction=ensemble_pred,
            confidence=confidence
        )

    def get_prediction_direction(self, prediction: ModelEnsemblePrediction) -> str:
        """Get trading direction from prediction"""
        if prediction.ensemble_prediction > 0.01:
            return "BUY"
        elif prediction.ensemble_prediction < -0.01:
            return "SELL"
        else:
            return "HOLD"
