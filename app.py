"""
Interactive Multi-Asset Momentum Strategy with Streamlit
=========================================================
Complete, corrected version with all features enabled
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
from simulation.monte_carlo import MonteCarloSimulator
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

# Advanced Analysis - ENABLED BY DEFAULT
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Advanced Analysis")

run_factor_analysis = st.sidebar.checkbox("Factor Exposure Analysis", value=True)
run_cost_analysis = st.sidebar.checkbox("Transaction Cost Analysis", value=True)

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
        
        # ====================================================================
        # MONTE CARLO SIMULATION SECTION
        # ====================================================================
        if run_monte_carlo:
            st.markdown("---")
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
        
        # ====================================================================
        # FACTOR ANALYSIS SECTION
        # ====================================================================
        if run_factor_analysis:
            st.markdown("---")
            st.header("📊 Factor Exposure Analysis")
            
            with st.spinner("Running factor analysis..."):
                try:
                    # Initialize factor analyzer
                    factor_analyzer = FactorAnalyzer(returns)
                    
                    # Model selection
                    factor_model = st.selectbox(
                        "Select Factor Model",
                        ['Carhart 4-Factor', 'Fama-French 3-Factor', 'Fama-French 5-Factor'],
                        index=0
                    )
                    
                    # Run analysis based on selection
                    if factor_model == 'Fama-French 3-Factor':
                        factor_results = factor_analyzer.fama_french_3factor()
                        model_desc = "Market, Size (SMB), Value (HML)"
                    elif factor_model == 'Carhart 4-Factor':
                        factor_results = factor_analyzer.carhart_4factor()
                        model_desc = "Market, Size (SMB), Value (HML), Momentum"
                    else:  # FF5
                        factor_results = factor_analyzer.fama_french_5factor()
                        model_desc = "Market, Size, Value, Profitability (RMW), Investment (CMA)"
                    
                    st.info(f"**Model**: {model_desc}")
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R-Squared", f"{factor_results.r_squared:.3f}")
                    col2.metric("Alpha (Annual)", f"{factor_results.alpha:.2%}")
                    col3.metric("Model Fit", 
                               "Strong" if factor_results.r_squared > 0.7 
                               else "Moderate" if factor_results.r_squared > 0.5 
                               else "Weak")
                    
                    # Factor loadings chart
                    st.subheader("Factor Loadings (Betas)")
                    fig_loadings = Visualizer.plot_factor_loadings(factor_results)
                    st.plotly_chart(fig_loadings, use_container_width=True)
                    
                    # Attribution chart
                    st.subheader("Return Attribution")
                    fig_attribution = Visualizer.plot_factor_attribution(factor_results)
                    st.plotly_chart(fig_attribution, use_container_width=True)
                    
                    # Interpretation guide
                    with st.expander("📖 How to Interpret Factor Loadings"):
                        st.markdown("""
                        **Factor loadings (betas) show your strategy's exposures:**
                        
                        - **Mkt-RF (Market)**: Beta to market excess returns
                          - β > 1.0: More volatile than market (aggressive)
                          - β < 1.0: Less volatile (defensive)
                          - β ≈ 0: Market neutral
                        
                        - **SMB (Size)**: Small vs Large cap exposure
                          - Positive: Small cap tilt
                          - Negative: Large cap tilt
                        
                        - **HML (Value)**: Value vs Growth exposure
                          - Positive: Value tilt
                          - Negative: Growth tilt
                        
                        - **Mom (Momentum)**: Momentum exposure
                          - Positive: Buys winners, sells losers
                          - Negative: Contrarian (buys losers)
                        
                        **Alpha**: Excess return not explained by factors (skill-based return)
                        
                        **R²**: % of returns explained by the model (higher = better fit)
                        """)
                    
                    # Rolling factor exposures
                    if st.checkbox("Show Time-Varying Factor Exposures"):
                        st.subheader("Rolling Factor Loadings")
                        
                        with st.spinner("Computing rolling 1-year factor exposures..."):
                            factor_data = factor_analyzer.fetch_factor_data(
                                ['Mkt-RF', 'SMB', 'HML', 'Mom', 'RF'] if factor_model == 'Carhart 4-Factor' 
                                else ['Mkt-RF', 'SMB', 'HML', 'RF']
                            )
                            rolling_betas = factor_analyzer.rolling_factor_loadings(factor_data, window=252)
                            
                            fig_rolling = Visualizer.plot_rolling_factor_loadings(rolling_betas)
                            st.plotly_chart(fig_rolling, use_container_width=True)
                    
                    # Style analysis
                    st.subheader("Investment Style Analysis")
                    style = factor_analyzer.style_analysis()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Market Beta", f"{style['Market Beta']:.3f}")
                        st.metric("Size Tilt (SMB)", f"{style['Size Tilt']:.3f}")
                        st.metric("Value Tilt (HML)", f"{style['Value Tilt']:.3f}")
                    with col2:
                        st.metric("Momentum Exposure", f"{style['Momentum Exposure']:.3f}")
                        st.info(f"**Style Classification:**\n\n{style['Style']}")
                
                except Exception as e:
                    st.error(f"❌ Error in factor analysis: {str(e)}")
                    st.exception(e)
        
        # ====================================================================
        # TRANSACTION COST ANALYSIS SECTION
        # ====================================================================
        if run_cost_analysis:
            st.markdown("---")
            st.header("💰 Transaction Cost Sensitivity Analysis")
            
            with st.spinner("Analyzing transaction costs..."):
                try:
                    # Initialize cost analyzer
                    cost_analyzer = TransactionCostAnalyzer(positions, prices, base_cost=transaction_cost)
                    
                    # Run sensitivity analysis
                    cost_results = cost_analyzer.sensitivity_analysis(
                        cost_levels=[i * 0.0005 for i in range(21)]  # 0 to 50 bps
                    )
                    
                    # Key metrics
                    st.subheader("Turnover Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    turnover_stats = cost_results.turnover_analysis.iloc[0]
                    col1.metric("Annual Turnover", f"{turnover_stats['Annualized Turnover']:.2f}")
                    col2.metric("Avg Daily Turnover", f"{turnover_stats['Average Daily Turnover']:.4f}")
                    col3.metric("Trading Frequency", f"{turnover_stats['Trading Frequency']:.2%}")
                    col4.metric("Breakeven Cost", f"{cost_results.breakeven_cost*10000:.1f} bps")
                    
                    # Cost sensitivity chart
                    st.subheader("Performance vs Transaction Costs")
                    fig_sensitivity = Visualizer.plot_cost_sensitivity(cost_results.cost_scenarios)
                    st.plotly_chart(fig_sensitivity, use_container_width=True)
                    
                    # Costs over time
                    st.subheader("Transaction Costs Over Time")
                    fig_turnover = Visualizer.plot_turnover_over_time(cost_results.cost_impact)
                    st.plotly_chart(fig_turnover, use_container_width=True)
                    
                    # Detailed analysis tabs
                    st.subheader("Detailed Cost Analysis")
                    cost_tab1, cost_tab2, cost_tab3 = st.tabs([
                        "Rebalancing Frequency", 
                        "Cost by Asset", 
                        "Slippage Impact"
                    ])
                    
                    with cost_tab1:
                        st.markdown("### Optimal Rebalancing Frequency")
                        st.info("Testing different rebalancing frequencies to balance costs vs tracking error...")
                        
                        rebal_df = cost_analyzer.optimal_rebalancing_frequency(
                            frequencies=[1, 5, 10, 20, 40, 60]
                        )
                        
                        fig_rebal = Visualizer.plot_rebalancing_frequency_analysis(rebal_df)
                        st.plotly_chart(fig_rebal, use_container_width=True)
                        
                        st.dataframe(rebal_df, use_container_width=True)
                        
                        # Recommendation
                        best_freq = rebal_df.loc[rebal_df['Sharpe_Ratio'].idxmax(), 'Rebalance_Frequency']
                        st.success(f"💡 **Recommendation**: Based on Sharpe ratio, optimal frequency is **{best_freq}**")

                    with cost_tab2:
                        st.markdown("### Transaction Costs by Asset")
                        
                        breakdown = cost_analyzer.cost_breakdown_by_asset()
                        

                        fig_breakdown = Visualizer.plot_cost_breakdown_by_asset(breakdown, top_n=10)
                        st.plotly_chart(fig_breakdown, use_container_width=True)
                        
                        st.dataframe(breakdown, use_container_width=True)

                    with cost_tab3:
                        st.markdown("### Market Impact & Slippage Analysis")
                        st.info("Estimating impact of market impact costs beyond fixed transaction fees...")
                        
                        slippage_df = cost_analyzer.slippage_impact_analysis(
                            slippage_levels=[0, 0.5, 1.0, 1.5, 2.0]
                        )
                        
                        st.dataframe(slippage_df, use_container_width=True)
                        
                        st.warning("""
                        **Note**: Market impact typically increases with trade size. Larger position changes 
                        experience more slippage. This analysis shows how returns degrade with increasing 
                        market impact costs.
                        """)
                    
                    # Cost efficiency threshold
                    with st.expander("🔍 Minimum Trade Threshold Analysis"):
                        st.markdown("### Impact of Trade Size Thresholds")
                        st.info("Testing minimum position change thresholds to reduce unnecessary rebalancing...")
                        
                        threshold_df = cost_analyzer.cost_efficient_threshold(
                            threshold_values=[0, 0.01, 0.02, 0.05, 0.10]
                        )
                        
                        st.dataframe(threshold_df, use_container_width=True)
                        
                        # Find optimal threshold
                        optimal_idx = threshold_df['Annual_Return'].idxmax()
                        optimal_threshold = threshold_df.loc[optimal_idx, 'Threshold_Pct']
                        st.success(f"💡 **Optimal Threshold**: {optimal_threshold:.1f}% minimizes costs while maintaining returns")
                
                except Exception as e:
                    st.error(f"❌ Error in transaction cost analysis: {str(e)}")
                    st.exception(e)
        
        # ====================================================================
        # DOWNLOAD RESULTS SECTION
        # ====================================================================
        st.markdown("---")
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
    5. **Enable Advanced Analysis**: Toggle Monte Carlo, Factor Analysis, Transaction Costs
    6. **Run Backtest**: Click the button to see results
    
    ### 📊 Strategy Overview
    
    This strategy combines:
    - **Time-Series Momentum**: Trades assets based on their own historical performance
    - **Cross-Sectional Momentum**: Ranks assets relative to each other
    - **Volatility-Based Position Sizing**: Allocates more to lower volatility assets
    - **Transaction Cost Modeling**: Realistic cost assumptions
    
    ### 🎯 Advanced Features
    
    - **Monte Carlo Simulation**: Project 10,000+ scenarios for risk assessment
    - **Factor Exposure Analysis**: Understand return drivers (Fama-French models)
    - **Transaction Cost Analysis**: Optimize rebalancing frequency and minimize costs
    """)
    
    # Show example configuration
    with st.expander("💡 Example Configurations"):
        st.markdown("""
        **Conservative Portfolio:**
        - Assets: SPY, AGG, GLD
        - Target Vol: 10%
        - Lookbacks: 60/120/180
        - Transaction Cost: 5 bps
        
        **Balanced Portfolio:**
        - Assets: SPY, QQQ, IWM, EFA, AGG, GLD
        - Target Vol: 15%
        - Lookbacks: 20/60/120
        - Transaction Cost: 10 bps
        
        **Aggressive Portfolio:**
        - Assets: QQQ, EEM, GC=F, CL=F
        - Target Vol: 20%
        - Lookbacks: 20/40/60
        - Transaction Cost: 15 bps
        
        **Multi-Asset Diversified:**
        - Assets: All available (Equities + FX + Futures)
        - Target Vol: 15%
        - Lookbacks: 20/60/120
        - Transaction Cost: 10 bps
        """)
    
    # Show key metrics explanation
    with st.expander("📚 Understanding Key Metrics"):
        st.markdown("""
        ### Performance Metrics
        
        - **Total Return**: Cumulative return over the entire backtest period
        - **Annual Return**: Annualized rate of return
        - **Sharpe Ratio**: Risk-adjusted return (return per unit of volatility)
        - **Annual Volatility**: Standard deviation of returns (annualized)
        - **Max Drawdown**: Largest peak-to-trough decline
        
        ### Factor Analysis
        
        - **Alpha**: Excess return not explained by factor exposures (skill)
        - **Beta**: Sensitivity to market movements
        - **R-Squared**: % of returns explained by the factor model
        - **Factor Loadings**: Exposure to different risk factors
        
        ### Transaction Costs
        
        - **Turnover**: How frequently the portfolio is rebalanced
        - **Breakeven Cost**: Maximum cost the strategy can tolerate
        - **Cost Sensitivity**: How performance changes with different cost levels
        - **Optimal Rebalancing**: Best frequency to balance costs vs performance
        
        ### Monte Carlo Simulation
        
        - **VaR (Value at Risk)**: Maximum expected loss at given confidence
        - **CVaR (Conditional VaR)**: Average loss beyond VaR threshold
        - **Probability Analysis**: Likelihood of different outcomes
        - **Percentiles**: Range of possible final values
        """)
    
    # Contact and support
    st.markdown("---")
    st.markdown("""
    ### 📧 Support
    
    For questions, issues, or feature requests:
    - Visit the [GitHub Repository](https://github.com/yourusername/momentum-strategy)
    - Report bugs in [Issues](https://github.com/yourusername/momentum-strategy/issues)
    - Read the [Documentation](https://github.com/yourusername/momentum-strategy/blob/main/README.md)
    
    ### ⚠️ Disclaimer
    
    This software is for educational and research purposes only. Not financial advice.
    Past performance does not guarantee future results. Trading involves risk of loss.
    """)
    
                    
                    
                        
                        
