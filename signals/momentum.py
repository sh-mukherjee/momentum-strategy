"""
Momentum signal generation
"""
import numpy as np
import pandas as pd
from typing import List

class MomentumSignals:
    """Generate momentum signals using multiple timeframes"""
    
    def __init__(self, lookback_periods: List[int]):
        self.lookback_periods = lookback_periods
    
    def calculate_returns(self, prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate period returns"""
        return prices.pct_change(period)
    
    def time_series_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Time-series momentum: long if positive return, short if negative
        Average across multiple lookback periods
        """
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        for period in self.lookback_periods:
            returns = self.calculate_returns(prices, period)
            signals += np.sign(returns)
        
        signals = signals / len(self.lookback_periods)
        return signals
    
    def cross_sectional_momentum(self, prices: pd.DataFrame, 
                                 lookback: int) -> pd.DataFrame:
        """
        Cross-sectional momentum: rank assets by returns
        """
        returns = self.calculate_returns(prices, lookback)
        ranks = returns.rank(axis=1, pct=True)
        
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        signals[ranks > 0.7] = 1
        signals[ranks < 0.3] = -1
        
        return signals
    
    def combined_signal(self, prices: pd.DataFrame, cs_lookback: int) -> pd.DataFrame:
        """Combine time-series and cross-sectional momentum"""
        ts_signal = self.time_series_momentum(prices)
        cs_signal = self.cross_sectional_momentum(prices, cs_lookback)
        combined = (ts_signal + cs_signal) / 2
        return combined
