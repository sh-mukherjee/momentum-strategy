"""
visualization/plots.py - Plotly version for interactive charts
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict
from scipy import stats  # For returns distribution

class Visualizer:
    """Create interactive performance visualizations using Plotly"""
    
    @staticmethod
    def plot_cumulative_returns_streamlit(returns: pd.Series, 
                                         benchmark_returns: pd.Series = None):
        """Plot cumulative returns with Plotly"""
        fig = go.Figure()
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Add strategy line
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode='lines',
            name='Strategy',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Date</b>: %{x}<br><b>Return</b>: %{y:.2%}<extra></extra>'
        ))
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            bench_cum = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>Date</b>: %{x}<br><b>Return</b>: %{y:.2%}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Cumulative Returns',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            yaxis=dict(tickformat='.0%')
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ])
            )
        )
        
        return fig
    
    @staticmethod
    def plot_drawdown_streamlit(returns: pd.Series):
        """Plot drawdown chart with Plotly"""
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        # Add drawdown area
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            mode='lines',
            name='Drawdown',
            line=dict(color='#d62728', width=2),
            fillcolor='rgba(214, 39, 40, 0.3)',
            hovertemplate='<b>Date</b>: %{x}<br><b>Drawdown</b>: %{y:.2%}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Drawdown Over Time',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Drawdown',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=False,
            yaxis=dict(tickformat='.0%')
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ])
            )
        )
        
        return fig
    
    @staticmethod
    def plot_rolling_sharpe_streamlit(returns: pd.Series, window: int):
        """Plot rolling Sharpe ratio with Plotly"""
        # Calculate rolling Sharpe
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        fig = go.Figure()
        
        # Add rolling Sharpe line
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color='#2ca02c', width=2.5),
            hovertemplate='<b>Date</b>: %{x}<br><b>Sharpe Ratio</b>: %{y:.2f}<extra></extra>'
        ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, 
                     annotation_text="Sharpe = 0")
        fig.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.5,
                     annotation_text="Sharpe = 1.0")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Rolling {window}-Day Sharpe Ratio',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Sharpe Ratio',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_position_heatmap(positions: pd.DataFrame):
        """Plot position allocation heatmap over time with Plotly"""
        # Sample positions to avoid overcrowding
        sample_freq = max(1, len(positions) // 100)
        positions_sample = positions.iloc[::sample_freq]
        
        # Format dates for better display
        date_labels = [d.strftime('%Y-%m-%d') for d in positions_sample.index]
        
        fig = go.Figure(data=go.Heatmap(
            z=positions_sample.T.values,
            x=date_labels,
            y=positions_sample.columns,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="Position<br>Weight"),
            hovertemplate='<b>Asset</b>: %{y}<br><b>Date</b>: %{x}<br><b>Weight</b>: %{z:.2%}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Asset Position Weights Over Time',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Assets',
            template='plotly_white',
            height=600,
            xaxis=dict(
                tickangle=-45,
                nticks=20
            )
        )
        
        return fig
    
    @staticmethod
    def plot_monthly_returns_heatmap(returns: pd.Series):
        """Plot monthly returns heatmap"""
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table: rows = years, columns = months
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=month_names,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Return (%)", tickformat='.0%'),
            hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Monthly Returns Heatmap',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Month',
            yaxis_title='Year',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_monte_carlo_paths(mc_results, n_paths_display: int = 100):
        """
        Plot Monte Carlo simulation paths with confidence intervals
        
        Args:
            mc_results: MonteCarloResults object
            n_paths_display: Number of individual paths to display
        """
        from simulation.monte_carlo import MonteCarloResults
        
        fig = go.Figure()
        
        # Sample random paths to display (avoid overcrowding)
        n_sims = mc_results.simulated_paths.shape[0]
        sample_indices = np.random.choice(n_sims, min(n_paths_display, n_sims), replace=False)
        
        # Plot individual paths (light gray, transparent)
        for idx in sample_indices:
            fig.add_trace(go.Scatter(
                y=mc_results.simulated_paths[idx],
                mode='lines',
                line=dict(color='lightgray', width=0.5),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Plot percentile bands
        colors = {5: '#d62728', 25: '#ff7f0e', 50: '#2ca02c', 75: '#ff7f0e', 95: '#d62728'}
        names = {5: '5th Percentile', 25: '25th Percentile', 50: 'Median', 
                75: '75th Percentile', 95: '95th Percentile'}
        
        for percentile in [95, 75, 50, 25, 5]:
            fig.add_trace(go.Scatter(
                y=mc_results.percentiles[percentile],
                mode='lines',
                name=names[percentile],
                line=dict(color=colors[percentile], width=2.5),
                hovertemplate=f'<b>{names[percentile]}</b><br>Value: %{{y:.3f}}<extra></extra>'
            ))
        
        # Add initial value line
        fig.add_hline(y=1.0, line_dash="dash", line_color="black", 
                     annotation_text="Initial Value", opacity=0.5)
        
        fig.update_layout(
            title=dict(
                text='Monte Carlo Simulation - Portfolio Paths',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Trading Days',
            yaxis_title='Portfolio Value',
            template='plotly_white',
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )
        
        return fig
    
    @staticmethod
    def plot_final_value_distribution(mc_results):
        """Plot histogram of final portfolio values"""
        from simulation.monte_carlo import MonteCarloResults
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=mc_results.final_values,
            nbinsx=50,
            name='Final Values',
            marker_color='#1f77b4',
            opacity=0.7,
            hovertemplate='<b>Value Range</b>: %{x:.3f}<br><b>Count</b>: %{y}<extra></extra>'
        ))
        
        # Add percentile lines
        for percentile, color, name in [(5, 'red', '5th'), (50, 'green', 'Median'), (95, 'blue', '95th')]:
            value = mc_results.percentiles[percentile][-1]
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{name}: {value:.3f}",
                annotation_position="top"
            )
        
        # Add initial value line
        fig.add_vline(x=1.0, line_dash="dot", line_color="black",
                     annotation_text="Initial", opacity=0.5)
        
        fig.update_layout(
            title=dict(
                text='Distribution of Final Portfolio Values',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Final Portfolio Value',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_var_cvar_chart(mc_results):
        """Plot VaR and CVaR comparison"""
        from simulation.monte_carlo import MonteCarloResults
        
        confidences = ['90', '95', '99']
        var_values = [mc_results.var_cvar[f'VaR_{c}'] * 100 for c in confidences]
        cvar_values = [mc_results.var_cvar[f'CVaR_{c}'] * 100 for c in confidences]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f'{c}%' for c in confidences],
            y=var_values,
            name='VaR',
            marker_color='#ff7f0e',
            text=[f'{v:.1f}%' for v in var_values],
            textposition='outside',
            hovertemplate='<b>VaR</b><br>Confidence: %{x}<br>Value: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            x=[f'{c}%' for c in confidences],
            y=cvar_values,
            name='CVaR',
            marker_color='#d62728',
            text=[f'{v:.1f}%' for v in cvar_values],
            textposition='outside',
            hovertemplate='<b>CVaR</b><br>Confidence: %{x}<br>Value: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Value at Risk (VaR) vs Conditional VaR (CVaR)',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Confidence Level',
            yaxis_title='Expected Loss (%)',
            template='plotly_white',
            height=400,
            barmode='group',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    @staticmethod
    def plot_method_comparison(comparison_results: Dict):
        """
        Compare results from different Monte Carlo methods
        
        Args:
            comparison_results: Dict from MonteCarloSimulator.compare_methods()
        """
        methods = list(comparison_results.keys())
        
        # Extract metrics for comparison
        expected_returns = [comparison_results[m].metrics['expected_return'] * 100 
                          for m in methods]
        volatilities = [comparison_results[m].metrics['annual_volatility'] * 100 
                       for m in methods]
        max_drawdowns = [comparison_results[m].metrics['avg_max_drawdown'] * 100 
                        for m in methods]
        prob_profits = [comparison_results[m].metrics['prob_profit'] * 100 
                       for m in methods]
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Expected Return', 'Volatility', 
                          'Avg Max Drawdown', 'Probability of Profit'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Expected Return
        fig.add_trace(
            go.Bar(x=methods, y=expected_returns, name='Expected Return',
                  marker_color='#2ca02c', text=[f'{v:.1f}%' for v in expected_returns],
                  textposition='outside'),
            row=1, col=1
        )
        
        # Volatility
        fig.add_trace(
            go.Bar(x=methods, y=volatilities, name='Volatility',
                  marker_color='#ff7f0e', text=[f'{v:.1f}%' for v in volatilities],
                  textposition='outside'),
            row=1, col=2
        )
        
        # Max Drawdown
        fig.add_trace(
            go.Bar(x=methods, y=max_drawdowns, name='Max Drawdown',
                  marker_color='#d62728', text=[f'{v:.1f}%' for v in max_drawdowns],
                  textposition='outside'),
            row=2, col=1
        )
        
        # Probability of Profit
        fig.add_trace(
            go.Bar(x=methods, y=prob_profits, name='Prob of Profit',
                  marker_color='#1f77b4', text=[f'{v:.1f}%' for v in prob_profits],
                  textposition='outside'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='Monte Carlo Method Comparison',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_white',
            height=700,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Probability (%)", row=2, col=2)
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series):
        """Plot returns distribution histogram"""
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=returns.values,
            nbinsx=50,
            name='Returns',
            marker_color='#1f77b4',
            opacity=0.7,
            hovertemplate='<b>Return Range</b>: %{x:.2%}<br><b>Count</b>: %{y}<extra></extra>'
        ))
        
        # Add normal distribution overlay
        from scipy import stats
        mu, sigma = returns.mean(), returns.std()
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x_range, mu, sigma) * len(returns) * (returns.max() - returns.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add mean line
        fig.add_vline(x=mu, line_dash="dot", line_color="green", 
                     annotation_text=f"Mean: {mu:.2%}")
        
        fig.update_layout(
            title=dict(
                text='Daily Returns Distribution',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400,
            showlegend=True,
            xaxis=dict(tickformat='.1%')
        )
        
        return fig
    
    # ========================================================================
    # FACTOR ANALYSIS VISUALIZATIONS
    # ========================================================================
    
    @staticmethod
    def plot_factor_loadings(results):
        """Plot factor loadings (betas) as bar chart"""
        from analysis.factor_exposure import FactorAnalysisResults
        
        fig = go.Figure()
        
        factors = results.factor_loadings.index.tolist()
        loadings = results.factor_loadings.values
        
        # Color based on positive/negative
        colors = ['#2ca02c' if l > 0 else '#d62728' for l in loadings]
        
        fig.add_trace(go.Bar(
            x=factors,
            y=loadings,
            marker_color=colors,
            text=[f'{l:.3f}' for l in loadings],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Loading: %{y:.4f}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title=dict(
                text=f'Factor Loadings (R² = {results.r_squared:.3f})',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Factor',
            yaxis_title='Loading (Beta)',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_factor_attribution(results):
        """Plot return attribution by factor"""
        from analysis.factor_exposure import FactorAnalysisResults
        
        fig = go.Figure()
        
        factors = results.attribution.index.tolist()
        contributions = results.attribution.values
        
        # Calculate percentages
        total = contributions.sum()
        percentages = (contributions / total * 100) if total != 0 else np.zeros_like(contributions)
        
        fig.add_trace(go.Bar(
            x=factors,
            y=contributions,
            marker_color='#1f77b4',
            text=[f'{c:.4f}<br>({p:.1f}%)' for c, p in zip(contributions, percentages)],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Contribution: %{y:.6f}<br>Percentage: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Return Attribution by Factor',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Factor',
            yaxis_title='Contribution to Returns',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_rolling_factor_loadings(rolling_betas: pd.DataFrame):
        """Plot time-varying factor exposures"""
        fig = go.Figure()
        
        # Plot each factor (excluding Alpha)
        factors = [col for col in rolling_betas.columns if col != 'Alpha']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, factor in enumerate(factors):
            fig.add_trace(go.Scatter(
                x=rolling_betas.index,
                y=rolling_betas[factor],
                mode='lines',
                name=factor,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{factor}</b><br>Date: %{{x}}<br>Loading: %{{y:.3f}}<extra></extra>'
            ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
        
        fig.update_layout(
            title=dict(
                text='Rolling Factor Exposures (1-Year Window)',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Factor Loading',
            template='plotly_white',
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )
        
        return fig
    
    # ========================================================================
    # TRANSACTION COST VISUALIZATIONS
    # ========================================================================
    
    @staticmethod
    def plot_cost_sensitivity(cost_scenarios: pd.DataFrame):
        """Plot performance metrics vs transaction costs"""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Annual Return vs Cost', 'Sharpe Ratio vs Cost',
                          'Total Costs Paid', 'Costs as % of Return'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Annual Return
        fig.add_trace(
            go.Scatter(x=cost_scenarios['Cost_BPS'], y=cost_scenarios['Annual_Return']*100,
                      mode='lines+markers', name='Annual Return',
                      line=dict(color='#2ca02c', width=2),
                      hovertemplate='Cost: %{x:.1f} bps<br>Return: %{y:.2f}%<extra></extra>'),
            row=1, col=1
        )
        
        # Sharpe Ratio
        fig.add_trace(
            go.Scatter(x=cost_scenarios['Cost_BPS'], y=cost_scenarios['Sharpe_Ratio'],
                      mode='lines+markers', name='Sharpe Ratio',
                      line=dict(color='#1f77b4', width=2),
                      hovertemplate='Cost: %{x:.1f} bps<br>Sharpe: %{y:.2f}<extra></extra>'),
            row=1, col=2
        )
        
        # Total Costs
        fig.add_trace(
            go.Scatter(x=cost_scenarios['Cost_BPS'], y=cost_scenarios['Total_Costs_Paid']*100,
                      mode='lines+markers', name='Total Costs',
                      line=dict(color='#d62728', width=2),
                      hovertemplate='Cost: %{x:.1f} bps<br>Total Paid: %{y:.2f}%<extra></extra>'),
            row=2, col=1
        )
        
        # Costs as % of Return
        fig.add_trace(
            go.Scatter(x=cost_scenarios['Cost_BPS'], y=cost_scenarios['Costs_Pct_of_Return'],
                      mode='lines+markers', name='Cost %',
                      line=dict(color='#ff7f0e', width=2),
                      hovertemplate='Cost: %{x:.1f} bps<br>% of Return: %{y:.1f}%<extra></extra>'),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Transaction Cost (basis points)", row=1, col=1)
        fig.update_xaxes(title_text="Transaction Cost (basis points)", row=1, col=2)
        fig.update_xaxes(title_text="Transaction Cost (basis points)", row=2, col=1)
        fig.update_xaxes(title_text="Transaction Cost (basis points)", row=2, col=2)
        
        fig.update_yaxes(title_text="Annual Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Total Costs Paid (%)", row=2, col=1)
        fig.update_yaxes(title_text="Costs as % of Return", row=2, col=2)
        
        fig.update_layout(
            title=dict(
                text='Transaction Cost Sensitivity Analysis',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_white',
            height=700,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_turnover_over_time(cost_impact: pd.DataFrame):
        """Plot turnover and costs over time"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Transaction Costs', 'Cumulative Costs'),
            shared_xaxes=True,
            vertical_spacing=0.12
        )
        
        # Daily costs
        fig.add_trace(
            go.Bar(x=cost_impact.index, y=cost_impact['Costs_Paid']*100,
                  name='Daily Costs', marker_color='#ff7f0e',
                  hovertemplate='Date: %{x}<br>Cost: %{y:.4f}%<extra></extra>'),
            row=1, col=1
        )
        
        # Cumulative costs
        fig.add_trace(
            go.Scatter(x=cost_impact.index, y=cost_impact['Cumulative_Costs']*100,
                      mode='lines', name='Cumulative Costs',
                      line=dict(color='#d62728', width=2),
                      fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.2)',
                      hovertemplate='Date: %{x}<br>Cumulative: %{y:.3f}%<extra></extra>'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Daily Cost (%)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Cost (%)", row=2, col=1)
        
        fig.update_layout(
            title=dict(
                text='Transaction Costs Over Time',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_white',
            height=600,
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_rebalancing_frequency_analysis(rebal_df: pd.DataFrame):
        """Plot optimal rebalancing frequency analysis"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Return vs Rebalancing Frequency', 'Sharpe vs Total Costs'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Return by frequency
        fig.add_trace(
            go.Bar(x=rebal_df['Rebalance_Frequency'], y=rebal_df['Annual_Return']*100,
                  marker_color='#2ca02c',
                  text=[f'{r:.2f}%' for r in rebal_df['Annual_Return']*100],
                  textposition='outside',
                  hovertemplate='Frequency: %{x}<br>Return: %{y:.2f}%<extra></extra>'),
            row=1, col=1
        )
        
        # Sharpe vs Costs scatter
        fig.add_trace(
            go.Scatter(x=rebal_df['Total_Costs']*100, y=rebal_df['Sharpe_Ratio'],
                      mode='markers+text',
                      marker=dict(size=12, color='#1f77b4'),
                      text=rebal_df['Rebalance_Frequency'],
                      textposition='top center',
                      hovertemplate='Costs: %{x:.3f}%<br>Sharpe: %{y:.2f}<extra></extra>'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Rebalancing Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Total Costs (%)", row=1, col=2)
        fig.update_yaxes(title_text="Annual Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        
        fig.update_layout(
            title=dict(
                text='Optimal Rebalancing Frequency Analysis',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_cost_breakdown_by_asset(breakdown_df: pd.DataFrame, top_n: int = 10):
        """Plot transaction costs by asset"""
        # Take top N assets by cost
        top_assets = breakdown_df.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_assets['Asset'],
            y=top_assets['Total_Costs']*100,
            marker_color='#ff7f0e',
            text=[f'{c:.4f}%' for c in top_assets['Total_Costs']*100],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Total Costs: %{y:.4f}%<br>Trade Freq: ' +
                         top_assets['Trade_Frequency'].astype(str) + '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Top {top_n} Assets by Transaction Costs',
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Asset',
            yaxis_title='Total Transaction Costs (%)',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    # Keep original methods for backward compatibility (non-Streamlit use)
    @staticmethod
    def plot_cumulative_returns(returns: pd.Series, benchmark_returns: pd.Series = None):
        """Original method for non-Streamlit scripts - shows figure"""
        fig = Visualizer.plot_cumulative_returns_streamlit(returns, benchmark_returns)
        fig.show()
    
    @staticmethod
    def plot_drawdown(returns: pd.Series):
        """Original method for non-Streamlit scripts - shows figure"""
        fig = Visualizer.plot_drawdown_streamlit(returns)
        fig.show()
    
    @staticmethod
    def plot_rolling_sharpe(returns: pd.Series, window: int):
        """Original method for non-Streamlit scripts - shows figure"""
        fig = Visualizer.plot_rolling_sharpe_streamlit(returns, window)
        fig.show()
