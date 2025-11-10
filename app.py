"""
Interactive Multi-Asset Momentum Strategy with Streamlit
=========================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import from our modules
from data import DataFetcher
from signals import MomentumSignals
from risk import RiskManager
from backtest import Backtester
from visualization import Visualizer
from simulation.monte_carlo import MonteCarloSimulator, print_simulation_summary
from analysis.factor_exposure import FactorAnalyzer
from analysis.transaction_costs import TransactionCostAnalyzer

# Page configuration
st.set_page_config(
    page_title="Momentum Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("🚀 Multi-Asset Momentum Strategy Backtester")
st.markdown("""
This interactive app allows you to customize and backtest a momentum trading strategy 
across equities, FX, and futures with real-time parameter adjustment.
""")

# ============================================================================
# SIDEBAR: User Inputs
# ============================================================================

st.sidebar.header("⚙️ Strategy Configuration")

# Date Range
st.sidebar.subheader("📅 Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2015, 1, 1),
        min_value=datetime(2010, 1, 1),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        min_value=start_date,
        max_value=datetime.now()
    )

# Asset Selection
st.sidebar.subheader("🎯 Asset Universe")

# Predefined asset groups
equity_options = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'DIA', 'VTI', 'AGG']
fx_options = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
futures_options = ['GC=F', 'CL=F', 'ZB=F', 'SI=F', 'NG=F']

selected_equities = st.sidebar.multiselect(
    "Equities",
    equity_options,
    default=['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ']
)

selected_fx = st.sidebar.multiselect(
    "FX Pairs",
    fx_options,
    default=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
)

selected_futures = st.sidebar.multiselect(
    "Futures",
    futures_options,
    default=['GC=F', 'CL=F', 'ZB=F']
)

# Custom tickers
custom_tickers = st.sidebar.text_input(
    "Additional Tickers (comma-separated)",
    placeholder="AAPL, MSFT, BTC-USD"
)

# Combine all selected tickers
all_tickers = selected_equities + selected_fx + selected_futures
if custom_tickers:
    custom_list = [t.strip() for t in custom_tickers.split(',')]
    all_tickers.extend(custom_list)

# Strategy Parameters
st.sidebar.subheader("📊 Momentum Parameters")

# Lookback periods
st.sidebar.markdown("**Time-Series Momentum Lookbacks (days)**")
lookback_short = st.sidebar.slider("Short-term", 5, 50, 20, 5)
lookback_medium = st.sidebar.slider("Medium-term", 30, 120, 60, 10)
lookback_long = st.sidebar.slider("Long-term", 60, 252, 120, 20)

lookback_periods = [lookback_short, lookback_medium, lookback_long]

# Cross-sectional lookback
cs_lookback = st.sidebar.slider(
    "Cross-Sectional Lookback (days)",
    20, 120, 60, 10
)

# Risk Management
st.sidebar.subheader("🛡️ Risk Management")

target_vol = st.sidebar.slider(
    "Target Volatility (%)",
    5.0, 30.0, 15.0, 1.0
) / 100

vol_lookback = st.sidebar.slider(
    "Volatility Lookback (days)",
    20, 120, 60, 10
)

# Transaction Costs
transaction_cost = st.sidebar.slider(
    "Transaction Cost (basis points)",
    0, 50, 10, 5
) / 10000

# Benchmark selection
benchmark_ticker = st.sidebar.selectbox(
    "Benchmark",
    ['SPY', 'QQQ', 'AGG', 'None'],
    index=0
)

# Monte Carlo Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Monte Carlo Simulation")

run_monte_carlo = st.sidebar.checkbox("Enable Monte Carlo Analysis", value=False)

if run_monte_carlo:
    mc_method = st.sidebar.selectbox(
        "Simulation Method",
        ['Historical Bootstrap', 'Parametric (Normal)', 'Geometric Brownian Motion', 'Compare All'],
        index=0
    )
    
    mc_simulations = st.sidebar.slider(
        "Number of Simulations",
        1000, 50000, 10000, 1000
    )
    
    mc_days = st.sidebar.slider(
        "Days to Simulate",
        30, 504, 252, 30
    )
    
    mc_paths_display = st.sidebar.slider(
        "Paths to Display",
        10, 500, 100, 10
    )

# ============================================================================
# RUN BACKTEST BUTTON
# ============================================================================

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if not all_tickers:
    st.warning("⚠️ Please select at least one asset to backtest.")
    st.stop()

if run_button:
    
    # Show selected configuration
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Assets Selected", len(all_tickers))
            st.metric("Date Range", f"{(end_date - start_date).days} days")
        with col2:
            st.metric("Target Volatility", f"{target_vol:.1%}")
            st.metric("Transaction Cost", f"{transaction_cost*10000:.0f} bps")
        with col3:
            st.metric("Momentum Lookbacks", f"{lookback_short}/{lookback_medium}/{lookback_long}")
            st.metric("Benchmark", benchmark_ticker if benchmark_ticker != 'None' else "N/A")
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch Data
        status_text.text("📥 Fetching market data...")
        progress_bar.progress(20)
        
        fetcher = DataFetcher(start_date.strftime('%Y-%m-%d'), 
                             end_date.strftime('%Y-%m-%d'))
        prices = fetcher.fetch_data(all_tickers)
        
        if prices.empty:
            st.error("❌ No data retrieved. Please check your ticker symbols and date range.")
            st.stop()
        
        # Step 2: Generate Signals
        status_text.text("🎯 Generating momentum signals...")
        progress_bar.progress(40)
        
        signal_gen = MomentumSignals(lookback_periods=lookback_periods)
        signals = signal_gen.combined_signal(prices, cs_lookback)
        
        # Step 3: Risk Management
        status_text.text("🛡️ Applying risk management...")
        progress_bar.progress(60)
        
        risk_mgr = RiskManager(target_vol=target_vol, vol_lookback=vol_lookback)
        positions = risk_mgr.volatility_scaled_weights(signals, prices)
        
        # Step 4: Backtest
        status_text.text("📊 Running backtest...")
        progress_bar.progress(80)
        
        backtester = Backtester(transaction_cost=transaction_cost)
        returns = backtester.calculate_portfolio_returns(positions, prices)
        
        # Get benchmark if selected
        benchmark_returns = None
        if benchmark_ticker != 'None' and benchmark_ticker in prices.columns:
            benchmark_returns = prices[benchmark_ticker].pct_change()
        
        # Calculate metrics
        strategy_metrics = backtester.calculate_metrics(returns)
        
        progress_bar.progress(100)
        status_text.text("✅ Backtest complete!")
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        
        st.success("✅ Backtest completed successfully!")
        
        # Performance Metrics
        st.header("📈 Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Parse metrics (remove % and convert back to float for display)
        total_ret = float(strategy_metrics['Total Return'].strip('%')) / 100
        annual_ret = float(strategy_metrics['Annual Return'].strip('%')) / 100
        annual_vol = float(strategy_metrics['Annual Volatility'].strip('%')) / 100
        sharpe = float(strategy_metrics['Sharpe Ratio'])
        max_dd = float(strategy_metrics['Max Drawdown'].strip('%')) / 100
        
        col1.metric("Total Return", f"{total_ret:.2%}")
        col2.metric("Annual Return", f"{annual_ret:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col4.metric("Volatility", f"{annual_vol:.2%}")
        col5.metric("Max Drawdown", f"{max_dd:.2%}")
        
        # Benchmark comparison (if available)
        if benchmark_returns is not None:
            st.subheader(f"📊 vs. Benchmark ({benchmark_ticker})")
            benchmark_metrics = backtester.calculate_metrics(benchmark_returns)
            
            bench_annual_ret = float(benchmark_metrics['Annual Return'].strip('%')) / 100
            bench_sharpe = float(benchmark_metrics['Sharpe Ratio'])
            bench_max_dd = float(benchmark_metrics['Max Drawdown'].strip('%')) / 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Excess Return",
                f"{(annual_ret - bench_annual_ret):.2%}",
                delta=f"{(annual_ret - bench_annual_ret):.2%}"
            )
            col2.metric(
                "Sharpe Difference",
                f"{(sharpe - bench_sharpe):.2f}",
                delta=f"{(sharpe - bench_sharpe):.2f}"
            )
            col3.metric(
                "Drawdown Difference",
                f"{(max_dd - bench_max_dd):.2%}",
                delta=f"{(max_dd - bench_max_dd):.2%}",
                delta_color="inverse"
            )
        
        # Visualizations
        st.header("📉 Performance Charts")
        
        # Cumulative returns
        st.subheader("Cumulative Returns")
        fig1 = Visualizer.plot_cumulative_returns_streamlit(returns, benchmark_returns)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Drawdown
        st.subheader("Drawdown Analysis")
        fig2 = Visualizer.plot_drawdown_streamlit(returns)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Rolling Sharpe
        st.subheader("Rolling Sharpe Ratio (1-Year)")
        fig3 = Visualizer.plot_rolling_sharpe_streamlit(returns, 252)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Additional charts in tabs
        tab1, tab2, tab3 = st.tabs(["Position Heatmap", "Monthly Returns", "Returns Distribution"])
        
        with tab1:
            fig4 = Visualizer.plot_position_heatmap(positions)
            st.plotly_chart(fig4, use_container_width=True)
        
        with tab2:
            fig5 = Visualizer.plot_monthly_returns_heatmap(returns)
            st.plotly_chart(fig5, use_container_width=True)
        
        with tab3:
            fig6 = Visualizer.plot_returns_distribution(returns)
            st.plotly_chart(fig6, use_container_width=True)
        
        # Monte Carlo Simulation Section
        if run_monte_carlo:
            st.header("🎲 Monte Carlo Risk Assessment")
            
            st.info("Running Monte Carlo simulation to project future portfolio outcomes...")
            
            # Initialize simulator with strategy returns
            mc_simulator = MonteCarloSimulator(returns, initial_value=1.0)
            
            # Run simulation based on selected method
            if mc_method == 'Compare All':
                st.subheader("Method Comparison")
                comparison_results = mc_simulator.compare_methods(
                    n_simulations=mc_simulations,
                    n_days=mc_days,
                    random_seed=42
                )
                
                # Show comparison chart
                fig_comparison = Visualizer.plot_method_comparison(comparison_results)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Show detailed results for each method
                with st.expander("📊 Detailed Results by Method"):
                    for method_name, mc_results in comparison_results.items():
                        st.markdown(f"### {method_name}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Expected Return", 
                                   f"{mc_results.metrics['expected_return']:.2%}")
                        col2.metric("Annual Volatility", 
                                   f"{mc_results.metrics['annual_volatility']:.2%}")
                        col3.metric("Prob of Profit", 
                                   f"{mc_results.metrics['prob_profit']:.2%}")
                        col4.metric("Avg Max DD", 
                                   f"{mc_results.metrics['avg_max_drawdown']:.2%}")
                        st.markdown("---")
                
                # Use Historical Bootstrap for detailed analysis
                mc_results = comparison_results['Historical Bootstrap']
                
            else:
                # Run single method
                if mc_method == 'Historical Bootstrap':
                    mc_results = mc_simulator.simulate_historical_bootstrap(
                        n_simulations=mc_simulations,
                        n_days=mc_days,
                        random_seed=42
                    )
                elif mc_method == 'Parametric (Normal)':
                    mc_results = mc_simulator.simulate_parametric(
                        n_simulations=mc_simulations,
                        n_days=mc_days,
                        random_seed=42
                    )
                else:  # GBM
                    mc_results = mc_simulator.simulate_gbm(
                        n_simulations=mc_simulations,
                        n_days=mc_days,
                        random_seed=42
                    )
            
            # Display key metrics
            st.subheader("📊 Simulation Summary")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric(
                "Expected Value",
                f"{mc_results.metrics['expected_value']:.3f}",
                delta=f"{mc_results.metrics['expected_return']:.2%}"
            )
            col2.metric(
                "Prob of Profit",
                f"{mc_results.metrics['prob_profit']:.1%}"
            )
            col3.metric(
                "VaR (95%)",
                f"{mc_results.var_cvar['VaR_95']:.2%}"
            )
            col4.metric(
                "CVaR (95%)",
                f"{mc_results.var_cvar['CVaR_95']:.2%}"
            )
            col5.metric(
                "Avg Max DD",
                f"{mc_results.metrics['avg_max_drawdown']:.2%}"
            )
            
            # Simulation paths visualization
            st.subheader("Projected Portfolio Paths")
            fig_paths = Visualizer.plot_monte_carlo_paths(mc_results, mc_paths_display)
            st.plotly_chart(fig_paths, use_container_width=True)
            
            # Create tabs for different MC visualizations
            mc_tab1, mc_tab2, mc_tab3 = st.tabs([
                "Final Value Distribution", 
                "Risk Metrics (VaR/CVaR)", 
                "Probability Analysis"
            ])
            
            with mc_tab1:
                fig_dist = Visualizer.plot_final_value_distribution(mc_results)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with mc_tab2:
                fig_var = Visualizer.plot_var_cvar_chart(mc_results)
                st.plotly_chart(fig_var, use_container_width=True)
                
                # Explanation
                st.info("""
                **Understanding VaR & CVaR:**
                - **VaR (Value at Risk)**: Maximum expected loss at a given confidence level
                - **CVaR (Conditional VaR)**: Average loss when VaR is exceeded (tail risk)
                
                Example: VaR(95%) = 10% means there's a 5% chance of losing more than 10%
                """)
            
            with mc_tab3:
                st.markdown("### Probability of Different Outcomes")
                
                prob_data = {
                    'Outcome': [
                        'Any Profit',
                        'Loss > 10%',
                        'Loss > 20%',
                        'Loss > 50%'
                    ],
                    'Probability': [
                        f"{mc_results.metrics['prob_profit']:.2%}",
                        f"{mc_results.metrics['prob_loss_10pct']:.2%}",
                        f"{mc_results.metrics['prob_loss_20pct']:.2%}",
                        f"{mc_results.metrics['prob_loss_50pct']:.2%}"
                    ]
                }
                
                st.table(pd.DataFrame(prob_data))
                
                # Percentiles table
                st.markdown("### Projected Final Values (Percentiles)")
                
                percentile_data = {
                    'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
                    'Final Value': [
                        f"{mc_results.percentiles[5][-1]:.3f}",
                        f"{mc_results.percentiles[25][-1]:.3f}",
                        f"{mc_results.percentiles[50][-1]:.3f}",
                        f"{mc_results.percentiles[75][-1]:.3f}",
                        f"{mc_results.percentiles[95][-1]:.3f}"
                    ],
                    'Return': [
                        f"{(mc_results.percentiles[5][-1] - 1):.2%}",
                        f"{(mc_results.percentiles[25][-1] - 1):.2%}",
                        f"{(mc_results.percentiles[50][-1] - 1):.2%}",
                        f"{(mc_results.percentiles[75][-1] - 1):.2%}",
                        f"{(mc_results.percentiles[95][-1] - 1):.2%}"
                    ]
                }
                
                st.table(pd.DataFrame(percentile_data))
            
            # Detailed metrics in expander
            with st.expander("📋 Complete Monte Carlo Metrics"):
                metrics_data = []
                for key, value in mc_results.metrics.items():
                    if isinstance(value, float):
                        if 'prob' in key or 'return' in key or 'drawdown' in key or 'volatility' in key:
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    metrics_data.append({
                        'Metric': key.replace('_', ' ').title(),
                        'Value': formatted_value
                    })
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # Detailed metrics table
        with st.expander("📋 Detailed Metrics Table"):
            metrics_df = pd.DataFrame({
                'Metric': list(strategy_metrics.keys()),
                'Value': list(strategy_metrics.values())
            })
            st.dataframe(metrics_df, use_container_width=True)
        
        # Download results
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download returns
            returns_csv = returns.to_csv()
            st.download_button(
                label="📥 Download Returns (CSV)",
                data=returns_csv,
                file_name=f"strategy_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download positions
            positions_csv = positions.to_csv()
            st.download_button(
                label="📥 Download Positions (CSV)",
                data=positions_csv,
                file_name=f"strategy_positions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"❌ Error running backtest: {str(e)}")
        st.exception(e)
        
else:
    # Show instructions when backtest hasn't been run
    st.info("👈 Configure your strategy parameters in the sidebar and click **Run Backtest** to begin.")
    
    st.markdown("""
    ### 📖 How to Use This App
    
    1. **Set Date Range**: Choose your backtest period
    2. **Select Assets**: Pick from equities, FX pairs, and futures
    3. **Configure Momentum**: Adjust lookback periods for signal generation
    4. **Set Risk Parameters**: Define target volatility and position sizing
    5. **Run Backtest**: Click the button to see results
    
    ### 📊 Strategy Overview
    
    This strategy combines:
    - **Time-Series Momentum**: Trades assets based on their own historical performance
    - **Cross-Sectional Momentum**: Ranks assets relative to each other
    - **Volatility-Based Position Sizing**: Allocates more to lower volatility assets
    - **Transaction Cost Modeling**: Realistic cost assumptions
    """)
    
    # Show example configuration
    with st.expander("💡 Example Configuration"):
        st.markdown("""
        **Conservative Portfolio:**
        - Assets: SPY, AGG, GLD
        - Target Vol: 10%
        - Lookbacks: 60/120/180
        
        **Aggressive Portfolio:**
        - Assets: QQQ, EEM, GC=F, CL=F
        - Target Vol: 20%
        - Lookbacks: 20/40/60
        """)
