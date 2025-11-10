"""
risk/manager.py - Risk management and position sizing (Fixed)
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
        # Calculate returns first
        returns = prices.pct_change()
        
        # Debugging
        print(f"\n[RiskManager] Calculating volatility...")
        print(f"Returns shape: {returns.shape}")
        print(f"Non-zero returns: {(returns != 0).sum().sum()}")
        
        # Rolling volatility
        vol = returns.rolling(window=self.vol_lookback, min_periods=20).std() * np.sqrt(252)
        
        print(f"Volatility calculated. Non-NaN count: {vol.notna().sum().sum()}")
        print(f"Sample volatility (first asset, last 5 days):")
        if len(vol.columns) > 0:
            print(vol[vol.columns[0]].tail())
        
        return vol
    
    def volatility_scaled_weights(self, signals: pd.DataFrame, 
                                  prices: pd.DataFrame) -> pd.DataFrame:
        """
        Scale positions inversely to volatility (risk parity)
        Higher vol assets get smaller positions
        """
        print(f"\n[RiskManager] Calculating position weights...")
        print(f"Signals shape: {signals.shape}")
        print(f"Prices shape: {prices.shape}")
        
        # Check if signals have actual values
        print(f"Non-zero signals: {(signals != 0).sum().sum()}")
        print(f"Signal sample (first 5 days):")
        print(signals.head())
        
        # Calculate volatility
        vol = self.calculate_volatility(prices)
        
        # Inverse volatility weighting
        inv_vol = 1 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        
        # Count valid inverse volatilities
        valid_inv_vol = inv_vol.notna().sum().sum()
        print(f"Valid inverse volatilities: {valid_inv_vol}")
        
        # Normalize so sum of absolute weights = 1
        inv_vol_sum = inv_vol.abs().sum(axis=1)
        
        # Avoid division by zero
        inv_vol_sum = inv_vol_sum.replace(0, np.nan)
        
        normalized_weights = inv_vol.div(inv_vol_sum, axis=0)
        
        print(f"Normalized weights non-NaN: {normalized_weights.notna().sum().sum()}")
        
        # Apply signal direction
        positions = signals * normalized_weights
        
        # Scale to target volatility
        positions = positions * self.target_vol
        
        # Fill NaN with 0 (for initial period where volatility can't be calculated)
        positions = positions.fillna(0)
        
        print(f"Final positions:")
        print(f"  Shape: {positions.shape}")
        print(f"  Non-zero: {(positions != 0).sum().sum()}")
        print(f"  Mean position size: {positions.abs().mean().mean():.4f}")
        print(f"  Sample (first 5 days):")
        print(positions.head())
        
        # Warning if all positions are zero
        if (positions == 0).all().all():
            print("⚠️  WARNING: All positions are ZERO!")
            print("Possible causes:")
            print("  1. Signals are all zero")
            print("  2. Not enough data to calculate volatility")
            print("  3. Volatility calculation returned all NaN")
        
        return positions


# Test function
def test_risk_manager():
    """Test the risk manager with sample data"""
    print("Testing RiskManager...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Sample prices with actual variation
    np.random.seed(42)
    prices = pd.DataFrame({
        'SPY': 100 * np.exp(np.random.randn(len(dates)).cumsum() * 0.01),
        'QQQ': 100 * np.exp(np.random.randn(len(dates)).cumsum() * 0.012),
    }, index=dates)
    
    # Sample signals (not all zero!)
    signals = pd.DataFrame({
        'SPY': np.random.choice([-1, 0, 1], len(dates)),
        'QQQ': np.random.choice([-1, 0, 1], len(dates)),
    }, index=dates)
    
    print(f"\nSample prices:")
    print(prices.head())
    print(f"\nSample signals:")
    print(signals.head())
    print(f"\nPrice returns:")
    print(prices.pct_change().head())
    
    # Create risk manager
    risk_mgr = RiskManager(target_vol=0.15, vol_lookback=60)
    
    # Calculate positions
    positions = risk_mgr.volatility_scaled_weights(signals, prices)
    
    print(f"\n\nFinal positions summary:")
    print(f"Shape: {positions.shape}")
    print(f"Mean absolute position: {positions.abs().mean().mean():.4f}")
    print(f"Non-zero positions: {(positions != 0).sum().sum()}/{positions.size}")


if __name__ == "__main__":
    test_risk_manager()
