"""
Backtesting engine
"""
import pandas as pd
from typing import Dict

class Backtester:
    """Backtest strategy with transaction costs"""
    
    def __init__(self, transaction_cost: float):
        self.transaction_cost = transaction_cost
    
    def calculate_portfolio_returns(self, positions: pd.DataFrame, 
                                    prices: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns including transaction costs"""
        asset_returns = prices.pct_change()
        position_changes = positions.diff().abs()
        
        gross_returns = (positions.shift(1) * asset_returns).sum(axis=1)
        costs = (position_changes * self.transaction_cost).sum(axis=1)
        net_returns = gross_returns - costs
        
        return net_returns
    
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Number of Trades': len(returns)
        }
