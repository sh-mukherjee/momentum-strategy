"""
signals/momentum.py - Momentum signal generation (Fixed with debugging)
"""
import numpy as np
import pandas as pd
from typing import List

class MomentumSignals:
    """Generate momentum signals using multiple timeframes"""
    
    def __init__(self, lookback_periods: List[int]):
        self.lookback_periods = lookback_periods
        print(f"[MomentumSignals] Initialized with lookbacks: {lookback_periods}")
    
    def calculate_returns(self, prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate period returns"""
        # Calculate returns over the period
        returns = prices.pct_change(period)
        
        # Debugging
        non_zero = (returns != 0).sum().sum()
        print(f"  Period {period}: {non_zero} non-zero returns")
        
        return returns
    
    def time_series_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Time-series momentum: long if positive return, short if negative
        Average across multiple lookback periods
        """
        print(f"\n[MomentumSignals] Calculating time-series momentum...")
        print(f"Input prices shape: {prices.shape}")
        print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        
        # Initialize signals dataframe
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        for period in self.lookback_periods:
            # Calculate returns for this period
            returns = self.calculate_returns(prices, period)
            
            # Sign of return: +1 if positive, -1 if negative, 0 if zero/NaN
            period_signals = np.sign(returns)
            
            # Add to cumulative signals
            signals = signals + period_signals
        
        # Average signal across periods
        signals = signals / len(self.lookback_periods)
        
        # Replace NaN with 0 (for initial periods)
        signals = signals.fillna(0)
        
        print(f"Time-series signals generated:")
        print(f"  Non-zero signals: {(signals != 0).sum().sum()}/{signals.size}")
        print(f"  Signal range: [{signals.min().min():.2f}, {signals.max().max():.2f}]")
        print(f"  Sample (first 5 days):")
        print(signals.head())
        
        return signals
    
    def cross_sectional_momentum(self, prices: pd.DataFrame, 
                                 lookback: int) -> pd.DataFrame:
        """
        Cross-sectional momentum: rank assets by returns
        """
        print(f"\n[MomentumSignals] Calculating cross-sectional momentum...")
        print(f"Lookback period: {lookback} days")
        
        # Calculate returns
        returns = self.calculate_returns(prices, lookback)
        
        print(f"Cross-sectional returns calculated:")
        print(f"  Non-NaN returns: {returns.notna().sum().sum()}")
        print(f"  Return range: [{returns.min().min():.6f}, {returns.max().max():.6f}]")
        
        # Rank returns cross-sectionally (higher rank = better performance)
        # pct=True gives percentile ranks (0 to 1)
        ranks = returns.rank(axis=1, pct=True)
        
        print(f"Ranks calculated:")
        print(f"  Non-NaN ranks: {ranks.notna().sum().sum()}")
        
        # Initialize signals
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        # Top 30% long (+1), bottom 30% short (-1), middle neutral (0)
        signals[ranks > 0.7] = 1
        signals[ranks < 0.3] = -1
        
        print(f"Cross-sectional signals generated:")
        print(f"  Long positions: {(signals == 1).sum().sum()}")
        print(f"  Short positions: {(signals == -1).sum().sum()}")
        print(f"  Neutral positions: {(signals == 0).sum().sum()}")
        
        return signals
    
    def combined_signal(self, prices: pd.DataFrame, cs_lookback: int) -> pd.DataFrame:
        """Combine time-series and cross-sectional momentum"""
        print(f"\n{'='*70}")
        print("GENERATING COMBINED MOMENTUM SIGNALS")
        print(f"{'='*70}")
        
        # Check input prices
        print(f"Input prices:")
        print(f"  Shape: {prices.shape}")
        print(f"  Columns: {list(prices.columns)}")
        print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"  Sample (first 5 days):")
        print(prices.head())
        
        # Check for price changes
        price_changes = prices.diff()
        print(f"\nPrice changes:")
        print(f"  Non-zero changes: {(price_changes != 0).sum().sum()}")
        print(f"  Sample (first 5 days):")
        print(price_changes.head())
        
        # If no price changes, prices are constant!
        if (price_changes == 0).all().all():
            print("❌ ERROR: All prices are constant (no changes)!")
            print("This will result in zero returns and zero signals.")
            # Return zero signals
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        # Generate signals
        ts_signal = self.time_series_momentum(prices)
        cs_signal = self.cross_sectional_momentum(prices, cs_lookback)
        
        # Combine (simple average)
        combined = (ts_signal + cs_signal) / 2
        
        print(f"\nCombined signals:")
        print(f"  Non-zero signals: {(combined != 0).sum().sum()}/{combined.size}")
        print(f"  Signal distribution:")
        print(f"    Positive: {(combined > 0).sum().sum()}")
        print(f"    Negative: {(combined < 0).sum().sum()}")
        print(f"    Zero: {(combined == 0).sum().sum()}")
        print(f"  Signal range: [{combined.min().min():.2f}, {combined.max().max():.2f}]")
        print(f"  Sample (first 10 days):")
        print(combined.head(10))
        print(f"{'='*70}\n")
        
        return combined


# Test function
def test_momentum_signals():
    """Test momentum signals with sample data"""
    print("Testing MomentumSignals...")
    
    # Create sample data with actual price variation
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    np.random.seed(42)
    prices = pd.DataFrame({
        'SPY': 100 * np.exp(np.random.randn(len(dates)).cumsum() * 0.01),
        'QQQ': 100 * np.exp(np.random.randn(len(dates)).cumsum() * 0.012),
        'IWM': 100 * np.exp(np.random.randn(len(dates)).cumsum() * 0.015),
    }, index=dates)
    
    print(f"Sample prices:")
    print(prices.head(10))
    print(f"\nPrice statistics:")
    print(prices.describe())
    
    # Create signal generator
    signal_gen = MomentumSignals(lookback_periods=[20, 60, 120])
    
    # Generate signals
    signals = signal_gen.combined_signal(prices, cs_lookback=60)
    
    print(f"\n\nFinal signals summary:")
    print(f"Shape: {signals.shape}")
    print(f"Non-zero: {(signals != 0).sum().sum()}")
    print(f"Mean signal: {signals.mean().mean():.4f}")


if __name__ == "__main__":
    test_momentum_signals()
