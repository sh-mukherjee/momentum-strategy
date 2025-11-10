"""
Main execution script for Multi-Asset Momentum Strategy
"""
import warnings
warnings.filterwarnings('ignore')

# Import from our modules
from data import DataFetcher
from signals import MomentumSignals
from risk import RiskManager
from backtest import Backtester
from visualization import Visualizer
import config

def main():
    """Run the complete momentum strategy backtest"""
    
    print("=" * 70)
    print("MULTI-ASSET MOMENTUM STRATEGY BACKTEST")
    print("=" * 70)
    print()
    
    # 1. Fetch data
    print("Step 1: Fetching market data...")
    tickers = config.EQUITIES + config.FX_PAIRS + config.FUTURES
    fetcher = DataFetcher(config.START_DATE, config.END_DATE)
    prices = fetcher.fetch_data(tickers)
    print()
    
    # 2. Generate signals
    print("Step 2: Generating momentum signals...")
    signal_gen = MomentumSignals(lookback_periods=config.MOMENTUM_LOOKBACKS)
    signals = signal_gen.combined_signal(prices, config.CROSS_SECTIONAL_LOOKBACK)
    print(f"Signals generated for {signals.shape[1]} assets")
    print()
    
    # 3. Apply risk management
    print("Step 3: Applying risk management and position sizing...")
    risk_mgr = RiskManager(
        target_vol=config.TARGET_VOLATILITY,
        vol_lookback=config.VOLATILITY_LOOKBACK
    )
    positions = risk_mgr.volatility_scaled_weights(signals, prices)
    print(f"Positions calculated (target vol: {config.TARGET_VOLATILITY:.0%})")
    print()
    
    # 4. Backtest
    print("Step 4: Running backtest with transaction costs...")
    backtester = Backtester(transaction_cost=config.TRANSACTION_COST)
    returns = backtester.calculate_portfolio_returns(positions, prices)
    
    # Calculate benchmark returns
    benchmark_returns = prices['SPY'].pct_change()
    
    print()
    print("=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    
    # Strategy metrics
    strategy_metrics = backtester.calculate_metrics(returns)
    print("\nStrategy Performance:")
    for key, value in strategy_metrics.items():
        print(f"  {key}: {value}")
    
    # Benchmark metrics
    benchmark_metrics = backtester.calculate_metrics(benchmark_returns)
    print("\nBenchmark Performance (SPY):")
    for key, value in benchmark_metrics.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    
    # 5. Visualizations
    print("\nGenerating visualizations...")
    viz = Visualizer()
    viz.plot_cumulative_returns(returns, benchmark_returns)
    viz.plot_drawdown(returns)
    viz.plot_rolling_sharpe(returns, config.ROLLING_SHARPE_WINDOW)
    
    print("\nBacktest complete!")
    
    return returns, positions, prices


if __name__ == "__main__":
    returns, positions, prices = main()
