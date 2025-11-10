"""
analysis/transaction_costs.py - Transaction cost sensitivity analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class TransactionCostResults:
    """Container for transaction cost analysis results"""
    cost_scenarios: pd.DataFrame  # Performance across cost levels
    breakeven_cost: float  # Maximum cost before returns turn negative
    turnover_analysis: pd.DataFrame  # Turnover statistics
    cost_impact: pd.DataFrame  # Cost impact by time period
    optimal_rebalance_freq: Dict  # Optimal rebalancing frequency


class TransactionCostAnalyzer:
    """
    Analyze impact of transaction costs on strategy performance
    
    Features:
    - Sensitivity analysis across cost levels
    - Turnover analysis
    - Optimal rebalancing frequency
    - Cost breakdown by asset
    - Slippage impact assessment
    """
    
    def __init__(self, 
                 positions: pd.DataFrame,
                 prices: pd.DataFrame,
                 base_cost: float = 0.001):
        """
        Initialize analyzer
        
        Args:
            positions: DataFrame of position weights over time
            prices: DataFrame of asset prices
            base_cost: Base transaction cost (default 10 bps)
        """
        self.positions = positions
        self.prices = prices
        self.base_cost = base_cost
        
        # Calculate returns
        self.asset_returns = prices.pct_change()
    
    def calculate_turnover(self) -> pd.DataFrame:
        """
        Calculate portfolio turnover metrics
        
        Turnover = sum of absolute position changes
        """
        # Position changes
        position_changes = self.positions.diff().abs()
        
        # Daily turnover
        daily_turnover = position_changes.sum(axis=1)
        
        # Cumulative turnover
        cumulative_turnover = daily_turnover.cumsum()
        
        # Create DataFrame
        turnover_df = pd.DataFrame({
            'Daily_Turnover': daily_turnover,
            'Cumulative_Turnover': cumulative_turnover
        }, index=self.positions.index)
        
        # Summary statistics
        stats = {
            'Total Turnover': cumulative_turnover.iloc[-1],
            'Average Daily Turnover': daily_turnover.mean(),
            'Median Daily Turnover': daily_turnover.median(),
            'Max Daily Turnover': daily_turnover.max(),
            'Days with Trades': (daily_turnover > 0.001).sum(),
            'Trading Frequency': (daily_turnover > 0.001).sum() / len(daily_turnover),
            'Annualized Turnover': daily_turnover.mean() * 252
        }
        
        return turnover_df, stats
    
    def sensitivity_analysis(self, 
                            cost_levels: List[float] = None) -> TransactionCostResults:
        """
        Test strategy performance across different transaction cost levels
        
        Args:
            cost_levels: List of cost levels to test (in decimal, e.g., 0.001 = 10 bps)
        """
        if cost_levels is None:
            # Default: 0 to 50 bps in 5 bp increments
            cost_levels = [i * 0.0005 for i in range(21)]
        
        results = []
        
        for cost in cost_levels:
            # Calculate returns with this cost level
            gross_returns = (self.positions.shift(1) * self.asset_returns).sum(axis=1)
            
            # Transaction costs
            position_changes = self.positions.diff().abs()
            costs = (position_changes * cost).sum(axis=1)
            
            # Net returns
            net_returns = gross_returns - costs
            
            # Performance metrics
            total_return = (1 + net_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(net_returns)) - 1
            volatility = net_returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            # Total costs paid
            total_costs = costs.sum()
            
            results.append({
                'Cost_BPS': cost * 10000,
                'Total_Return': total_return,
                'Annual_Return': annual_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe,
                'Total_Costs_Paid': total_costs,
                'Costs_Pct_of_Return': (total_costs / total_return * 100) if total_return > 0 else np.nan
            })
        
        cost_scenarios = pd.DataFrame(results)
        
        # Find breakeven cost (where returns turn negative)
        breakeven = cost_scenarios[cost_scenarios['Total_Return'] > 0]['Cost_BPS'].max() if any(cost_scenarios['Total_Return'] > 0) else 0
        
        # Turnover analysis
        turnover_df, turnover_stats = self.calculate_turnover()
        
        # Cost impact over time
        position_changes = self.positions.diff().abs()
        cost_impact = pd.DataFrame({
            'Period': self.positions.index,
            'Costs_Paid': (position_changes * self.base_cost).sum(axis=1),
            'Cumulative_Costs': (position_changes * self.base_cost).sum(axis=1).cumsum()
        }).set_index('Period')
        
        return TransactionCostResults(
            cost_scenarios=cost_scenarios,
            breakeven_cost=breakeven / 10000,  # Convert to decimal
            turnover_analysis=pd.DataFrame([turnover_stats]),
            cost_impact=cost_impact,
            optimal_rebalance_freq={}  # To be filled by optimization
        )
    
    def optimal_rebalancing_frequency(self,
                                     frequencies: List[int] = None) -> pd.DataFrame:
        """
        Find optimal rebalancing frequency to balance costs and tracking error
        
        Args:
            frequencies: List of rebalancing periods (in days) to test
        """
        if frequencies is None:
            frequencies = [1, 5, 10, 20, 40, 60]  # Daily, weekly, biweekly, monthly, etc.
        
        results = []
        
        for freq in frequencies:
            # Resample positions to this frequency
            rebalanced_positions = self.positions.iloc[::freq].copy()
            rebalanced_positions = rebalanced_positions.reindex(
                self.positions.index, method='ffill'
            )
            
            # Calculate returns
            gross_returns = (rebalanced_positions.shift(1) * self.asset_returns).sum(axis=1)
            
            # Transaction costs (only on rebalance days)
            position_changes = rebalanced_positions.diff().abs()
            costs = (position_changes * self.base_cost).sum(axis=1)
            
            net_returns = gross_returns - costs
            
            # Tracking error vs daily rebalancing
            daily_returns = (self.positions.shift(1) * self.asset_returns).sum(axis=1)
            tracking_error = (net_returns - daily_returns).std() * np.sqrt(252)
            
            # Metrics
            total_return = (1 + net_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(net_returns)) - 1
            sharpe = (annual_return / (net_returns.std() * np.sqrt(252))) if net_returns.std() > 0 else 0
            total_costs = costs.sum()
            num_rebalances = (position_changes.sum(axis=1) > 0.001).sum()
            
            results.append({
                'Rebalance_Frequency': f'{freq}d',
                'Rebalances_Per_Year': 252 / freq,
                'Total_Rebalances': num_rebalances,
                'Annual_Return': annual_return,
                'Sharpe_Ratio': sharpe,
                'Total_Costs': total_costs,
                'Tracking_Error': tracking_error,
                'Cost_Benefit_Ratio': annual_return / total_costs if total_costs > 0 else np.inf
            })
        
        return pd.DataFrame(results)
    
    def cost_breakdown_by_asset(self) -> pd.DataFrame:
        """
        Analyze which assets generate most transaction costs
        """
        position_changes = self.positions.diff().abs()
        
        costs_by_asset = (position_changes * self.base_cost).sum()
        
        # Add statistics
        breakdown = pd.DataFrame({
            'Asset': costs_by_asset.index,
            'Total_Costs': costs_by_asset.values,
            'Avg_Trade_Size': position_changes.mean().values,
            'Trade_Frequency': (position_changes > 0.001).sum().values,
            'Cost_Per_Trade': costs_by_asset.values / ((position_changes > 0.001).sum().values + 1e-10)
        })
        
        breakdown = breakdown.sort_values('Total_Costs', ascending=False)
        
        # Add percentage
        breakdown['Pct_of_Total_Costs'] = (breakdown['Total_Costs'] / breakdown['Total_Costs'].sum()) * 100
        
        return breakdown
    
    def slippage_impact_analysis(self,
                                slippage_levels: List[float] = None) -> pd.DataFrame:
        """
        Estimate impact of market impact/slippage beyond fixed costs
        
        Slippage typically increases with trade size:
        Slippage = base_slippage + size_factor * trade_size
        
        Args:
            slippage_levels: List of slippage multipliers to test
        """
        if slippage_levels is None:
            slippage_levels = [0, 0.5, 1.0, 1.5, 2.0]  # Multiples of base cost
        
        results = []
        
        for slip_mult in slippage_levels:
            # Calculate variable slippage based on trade size
            position_changes = self.positions.diff().abs()
            
            # Slippage increases with square root of trade size (market impact model)
            slippage_costs = self.base_cost * slip_mult * np.sqrt(position_changes)
            total_slippage = slippage_costs.sum(axis=1)
            
            # Fixed costs
            fixed_costs = (position_changes * self.base_cost).sum(axis=1)
            
            # Total costs
            total_costs = fixed_costs + total_slippage
            
            # Net returns
            gross_returns = (self.positions.shift(1) * self.asset_returns).sum(axis=1)
            net_returns = gross_returns - total_costs
            
            # Metrics
            total_return = (1 + net_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(net_returns)) - 1
            
            results.append({
                'Slippage_Multiple': slip_mult,
                'Total_Fixed_Costs': fixed_costs.sum(),
                'Total_Slippage_Costs': total_slippage.sum(),
                'Total_All_Costs': total_costs.sum(),
                'Annual_Return': annual_return,
                'Return_Degradation': annual_return - results[0]['Annual_Return'] if len(results) > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def cost_efficient_threshold(self,
                                 threshold_values: List[float] = None) -> pd.DataFrame:
        """
        Test impact of minimum trade thresholds to reduce unnecessary rebalancing
        
        Args:
            threshold_values: List of minimum position change thresholds
        """
        if threshold_values is None:
            threshold_values = [0, 0.01, 0.02, 0.05, 0.10]  # 0% to 10%
        
        results = []
        
        for threshold in threshold_values:
            # Apply threshold: only rebalance if change > threshold
            position_changes = self.positions.diff()
            filtered_changes = position_changes.copy()
            filtered_changes[position_changes.abs() < threshold] = 0
            
            # Reconstruct positions with threshold
            filtered_positions = self.positions.iloc[0:1].copy()
            for i in range(1, len(self.positions)):
                new_pos = filtered_positions.iloc[-1] + filtered_changes.iloc[i]
                filtered_positions = pd.concat([filtered_positions, new_pos.to_frame().T])
            
            # Calculate returns
            gross_returns = (filtered_positions.shift(1) * self.asset_returns).sum(axis=1)
            
            # Costs (only on actual trades)
            actual_changes = filtered_changes.abs()
            costs = (actual_changes * self.base_cost).sum(axis=1)
            
            net_returns = gross_returns - costs
            
            # Metrics
            total_return = (1 + net_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(net_returns)) - 1
            total_costs = costs.sum()
            num_trades = (actual_changes.sum(axis=1) > 0.001).sum()
            
            results.append({
                'Threshold_Pct': threshold * 100,
                'Total_Trades': num_trades,
                'Total_Costs': total_costs,
                'Annual_Return': annual_return,
                'Cost_Savings': results[0]['Total_Costs'] - total_costs if len(results) > 0 else 0,
                'Return_Impact': annual_return - results[0]['Annual_Return'] if len(results) > 0 else 0
            })
        
        return pd.DataFrame(results)


def print_transaction_cost_summary(results: TransactionCostResults):
    """Pretty print transaction cost analysis"""
    
    print("=" * 70)
    print("TRANSACTION COST ANALYSIS")
    print("=" * 70)
    print()
    
    print("TURNOVER STATISTICS:")
    for key, value in results.turnover_analysis.iloc[0].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    print("COST SENSITIVITY:")
    print(f"  Breakeven Cost: {results.breakeven_cost*10000:.1f} basis points")
    print()
    
    print("PERFORMANCE AT DIFFERENT COST LEVELS:")
    print(results.cost_scenarios.to_string(index=False))
    print()
    
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Sample positions
    assets = ['SPY', 'QQQ', 'IWM']
    positions = pd.DataFrame(
        np.random.uniform(-0.3, 0.3, (len(dates), len(assets))),
        index=dates,
        columns=assets
    )
    
    # Sample prices
    prices = pd.DataFrame(
        np.cumsum(np.random.normal(0.0005, 0.01, (len(dates), len(assets))), axis=0) + 100,
        index=dates,
        columns=assets
    )
    
    # Run analysis
    analyzer = TransactionCostAnalyzer(positions, prices, base_cost=0.001)
    results = analyzer.sensitivity_analysis()
    
    print_transaction_cost_summary(results)
