import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import linregress
import ta

class FlashCrashAnomalyDetector:
    \"\"\"
    Anomaly detection engine based on the features explored in the 'surpriver' repository,
    adapted for short-term Flash Crash prediction in our system.
    \"\"\"
    def __init__(self, history_to_use=20, n_estimators=150, random_state=42):
        self.HISTORY_TO_USE = history_to_use
        self.detector = IsolationForest(n_estimators=n_estimators, contamination=0.01, random_state=random_state)
        self.is_fitted = False

    def calculate_slope(self, data):
        if len(data) < 2:
            return 0.0, 0.0, 1.0
        x_axis = np.arange(len(data))
        regression_model = linregress(x_axis, data)
        # Using handle for NaNs
        slope = round(regression_model.slope, 3) if not np.isnan(regression_model.slope) else 0.0
        r_value = round(abs(regression_model.rvalue), 3) if not np.isnan(regression_model.rvalue) else 0.0
        p_value = round(regression_model.pvalue, 4) if not np.isnan(regression_model.pvalue) else 1.0
        return slope, r_value, p_value

    def get_technical_indicators(self, price_data):
        \"\"\"
        Extracts slopes and technical indicators from a price DataFrame.
        Required columns: 'Open', 'High', 'Low', 'Close', 'Volume'
        \"\"\"
        features = []
        
        # Ensure we have enough data
        if len(price_data) < self.HISTORY_TO_USE + 5:
            return None

        # 1. RSI
        rsi = ta.momentum.RSIIndicator(price_data['Close'], window=14, fillna=True).rsi().values.tolist()
        slope_rsi, _, _ = self.calculate_slope(rsi[-self.HISTORY_TO_USE:])
        features.append(slope_rsi)
        features.append(rsi[-1])

        # 2. Accumulation Distribution (A/D)
        acc_dist = ta.volume.acc_dist_index(price_data['High'], price_data['Low'], price_data['Close'], price_data['Volume'], fillna=True).values.tolist()
        slope_acc_dist, _, _ = self.calculate_slope(acc_dist[-self.HISTORY_TO_USE:])
        features.append(slope_acc_dist)

        # 3. Commodity Channel Index (CCI)
        cci = ta.trend.cci(price_data['High'], price_data['Low'], price_data['Close'], window=20, constant=0.015, fillna=True).values.tolist()
        slope_cci, _, _ = self.calculate_slope(cci[-self.HISTORY_TO_USE:])
        features.append(slope_cci)
        features.append(cci[-1])

        # 4. Volume returns and changes
        vol = price_data['Volume'].values
        vol_safe = np.where(vol == 0, 1, vol) # avoid division by zero
        volume_returns = [vol_safe[i] / vol_safe[i-1] for i in range(1, len(vol_safe))]
        slope_vol, _, _ = self.calculate_slope(volume_returns[-self.HISTORY_TO_USE:])
        features.append(slope_vol)
        
        # 5. Volatility (std over last N bars)
        close_prices = price_data['Close'].values[-self.HISTORY_TO_USE:]
        volatility = np.std(close_prices)
        features.append(volatility)
        
        return features

    def fit(self, historical_data_list):
        \"\"\"
        Fits the Isolation Forest on a list of DataFrames (e.g., historical windows).
        \"\"\"
        all_features = []
        for df in historical_data_list:
            feats = self.get_technical_indicators(df)
            if feats is not None:
                all_features.append(feats)
                
        if len(all_features) > 0:
            X = np.array(all_features)
            # Impute NaNs if any exist before fitting
            X = np.nan_to_num(X, nan=0.0)
            self.detector.fit(X)
            self.is_fitted = True
            print(f\"Fit AnomalyDetector on {len(X)} samples.\")
        else:
            print(\"Not enough data to fit the anomaly detector.\")

    def predict(self, current_data):
        \"\"\"
        Returns the anomaly score. Negative score = Anomaly.
        \"\"\"
        if not self.is_fitted:
            raise ValueError(\"Model must be fitted first.\")
            
        feats = self.get_technical_indicators(current_data)
        if feats is None:
            return 0.0 # Neutral if not enough data
            
        X = np.array([feats])
        X = np.nan_to_num(X, nan=0.0)
        
        # decision_function gives score > 0 for inliers and < 0 for anomalies.
        score = self.detector.decision_function(X)[0]
        return score
