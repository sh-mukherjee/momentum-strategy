"""
backtest/engine.py - Backtesting engine with improved debugging
"""
import pandas as pd
import numpy as np
from typing import Dict

class Backtester:
    """Backtest strategy with transaction costs"""
    
    def __init__(self, transaction_cost: float):
        self.transaction_cost = transaction_cost
    
    def calculate_portfolio_returns(self, positions: pd.DataFrame, 
                                    prices: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns including transaction costs
        
        Args:
            positions: DataFrame of position weights (same index/columns as prices)
            prices: DataFrame of asset prices
            
        Returns:
            Series of daily portfolio returns
        """
        # Debugging: Check inputs
        print("\n" + "="*70)
        print("BACKTEST ENGINE - DEBUGGING")
        print("="*70)
        print(f"Positions shape: {positions.shape}")
        print(f"Prices shape: {prices.shape}")
        print(f"Positions date range: {positions.index[0]} to {positions.index[-1]}")
        print(f"Prices date range: {prices.index[0]} to {prices.index[-1]}")
        
        # Ensure indices are aligned
        common_dates = positions.index.intersection(prices.index)
        if len(common_dates) < len(positions):
            print(f"⚠️  Warning: Only {len(common_dates)}/{len(positions)} dates align")
        
        positions_aligned = positions.loc[common_dates]
        prices_aligned = prices.loc[common_dates]
        
        # Calculate asset returns (percentage change)
        asset_returns = prices_aligned.pct_change()
        
        # Debugging: Check returns calculation
        print(f"\nAsset returns shape: {asset_returns.shape}")
        print(f"Asset returns non-zero count: {(asset_returns != 0).sum().sum()}")
        print(f"Asset returns sample (first 5 days):")
        print(asset_returns.head())
        print(f"\nAsset returns statistics:")
        print(asset_returns.describe())
        
        # Check if returns are all zero
        if (asset_returns == 0).all().all():
            print("❌ ERROR: All asset returns are ZERO!")
            print("This means prices are not changing. Checking prices...")
            print("\nPrice changes (first 5 days):")
            print(prices_aligned.diff().head())
            print("\nPrices (first 5 days):")
            print(prices_aligned.head())
        
        # Position changes (for transaction cost calculation)
        position_changes = positions_aligned.diff().abs()
        
        # Debugging: Check positions
        print(f"\nPositions sample (first 5 days):")
        print(positions_aligned.head())
        print(f"Position changes non-zero count: {(position_changes != 0).sum().sum()}")
        
        # Gross returns (before costs)
        # Use t-1 positions with t returns
        lagged_positions = positions_aligned.shift(1)
        
        # Calculate returns for each asset
        asset_contribution = lagged_positions * asset_returns
        
        # Debugging: Check contributions
        print(f"\nAsset contribution shape: {asset_contribution.shape}")
        print(f"Asset contribution sample (first 5 days):")
        print(asset_contribution.head())
        
        # Sum across assets to get portfolio returns
        gross_returns = asset_contribution.sum(axis=1)
        
        print(f"\nGross returns non-zero count: {(gross_returns != 0).sum()}")
        print(f"Gross returns sample (first 10 days):")
        print(gross_returns.head(10))
        
        # Transaction costs
        costs = (position_changes * self.transaction_cost).sum(axis=1)
        
        print(f"\nTransaction costs sum: {costs.sum():.6f}")
        print(f"Avg daily cost: {costs.mean():.6f}")
        
        # Net returns
        net_returns = gross_returns - costs
        
        # Remove NaN from first day
        net_returns = net_returns.fillna(0)
        
        # Final check
        print(f"\nFinal net returns:")
        print(f"  Non-zero days: {(net_returns != 0).sum()}/{len(net_returns)}")
        print(f"  Mean daily return: {net_returns.mean():.6f}")
        print(f"  Std daily return: {net_returns.std():.6f}")
        print(f"  Total return: {(1 + net_returns).prod() - 1:.6f}")
        print("="*70 + "\n")
        
        return net_returns
    
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        
        # Handle edge cases
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Remove any NaN or inf values
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        
        # Annualization factor
        years = len(returns) / 252
        if years > 0:
            annual_return = (1 + total_return) ** (1/years) - 1
        else:
            annual_return = 0
        
        annual_vol = returns.std() * np.sqrt(252)
        
        if annual_vol != 0:
            sharpe = annual_return / annual_vol
        else:
            sharpe = 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Check for suspicious values
        if total_return == 0 and returns.std() == 0:
            print("⚠️  WARNING: Zero returns detected in metrics calculation!")
            print(f"Returns length: {len(returns)}")
            print(f"Returns sample: {returns.head()}")
        
        return {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Number of Trades': len(returns)
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics for error cases"""
        return {
            'Total Return': "0.00%",
            'Annual Return': "0.00%",
            'Annual Volatility': "0.00%",
            'Sharpe Ratio': "0.00",
            'Max Drawdown': "0.00%",
            'Number of Trades': 0
        }


# Test function
def test_backtester():
    """Test the backtester with sample data"""
    print("Testing Backtester...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Sample prices (should have daily changes)
    prices = pd.DataFrame({
        'SPY': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
        'QQQ': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.012),
    }, index=dates)
    
    # Sample positions
    positions = pd.DataFrame({
        'SPY': 0.5,
        'QQQ': 0.5,
    }, index=dates)
    
    print(f"\nSample prices (first 5 days):")
    print(prices.head())
    print(f"\nPrice changes (first 5 days):")
    print(prices.pct_change().head())
    
    # Run backtest
    backtester = Backtester(transaction_cost=0.001)
    returns = backtester.calculate_portfolio_returns(positions, prices)
    
    print(f"\nReturns calculated:")
    print(returns.head(10))
    
    # Calculate metrics
    metrics = backtester.calculate_metrics(returns)
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_backtester()
