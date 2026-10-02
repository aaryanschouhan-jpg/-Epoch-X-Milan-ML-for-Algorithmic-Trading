import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class Model:
  """Quant ML Trading Strategy for AlphaForge Trading Competition.

  Uses Causal Feature Engineering + Cross-Sectional Ridge Regression.
  """

  def __init__(self):
    self.model = Ridge(alpha=100.0)
    self.feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "vol_20",
        "zscore_20",
        "vol_ratio",
    ]

  def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    # 1. Price Momentum (1, 5, 10-period lagged returns)
    df["ret_1"] = df.groupby("asset_id")["close"].pct_change(1)
    df["ret_5"] = df.groupby("asset_id")["close"].pct_change(5)
    df["ret_10"] = df.groupby("asset_id")["close"].pct_change(10)

    # 2. Moving Average & Volatility Z-score
    df["ma_20"] = (
        df.groupby("asset_id")["close"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["vol_20"] = (
        df.groupby("asset_id")["close"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["zscore_20"] = (df["close"] - df["ma_20"]) / (df["vol_20"] + 1e-8)

    # 3. Volume Trend Ratio
    df["vol_ma_10"] = (
        df.groupby("asset_id")["volume"]
        .rolling(10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["vol_ratio"] = df["volume"] / (df["vol_ma_10"] + 1e-8)

    return df

  def train(self, X: pd.DataFrame, y=None):
    """Train the Ridge model on historical returns."""
    df_feat = self._extract_features(X)

    # Target: Next period percentage return
    df_feat["target"] = (
        df_feat.groupby("asset_id")["close"].pct_change().shift(-1)
    )

    # Clean NaNs
    df_clean = df_feat.dropna(subset=self.feature_cols + ["target"])

    X_train = df_clean[self.feature_cols]
    y_train = df_clean["target"]

    self.model.fit(X_train, y_train)

  def predict(self, X: pd.DataFrame) -> np.ndarray:
    """Generates continuous signals in [-1.0, 1.0] for the current timestep."""
    current_time = X["timestamp"].max()

    # Compute features up to current time
    df_feat = self._extract_features(X)

    # Filter for latest step and sort by asset_id
    current = df_feat[df_feat["timestamp"] == current_time].sort_values(
        "asset_id"
    )

    X_curr = current[self.feature_cols].fillna(0)
    raw_preds = self.model.predict(X_curr)

    # Cross-sectional demean & normalize signal across all active assets
    demeaned = raw_preds - np.mean(raw_preds)
    std_val = np.std(demeaned)
    if std_val > 0:
      norm_signals = demeaned / std_val
    else:
      norm_signals = demeaned

    # Bound strictly between [-1.0, 1.0]
    signals = np.clip(norm_signals * 0.4, -1.0, 1.0)
    return signals
