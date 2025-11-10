"""
Risk management and position sizing
"""
import numpy as np
import pandas as pd

class RiskManager:
    """Volatility-based position sizing and risk controls"""
    
    def __init__(self, target_vol: float, vol_lookback: int):
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
    
    def calculate_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling realized volatility"""
        returns = prices.pct_change()
        vol = returns.rolling(window=self.vol_lookback).std() * np.sqrt(252)
        return vol
    
    def volatility_scaled_weights(self, signals: pd.DataFrame, 
                                  prices: pd.DataFrame) -> pd.DataFrame:
        """
        Scale positions inversely to volatility (risk parity)
        """
        vol = self.calculate_volatility(prices)
        
        # Inverse volatility weighting
        inv_vol = 1 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        
        # Normalize
        inv_vol_sum = inv_vol.abs().sum(axis=1)
        normalized_weights = inv_vol.div(inv_vol_sum, axis=0)
        
        # Apply signal direction
        positions = signals * normalized_weights
        
        # Scale to target volatility
        positions = positions * self.target_vol
        
        return positions.fillna(0)
