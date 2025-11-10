"""
analysis/factor_exposure.py - Factor exposure and attribution analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FactorAnalysisResults:
    """Container for factor analysis results"""
    factor_loadings: pd.Series  # Beta coefficients for each factor
    factor_returns: pd.DataFrame  # Historical factor returns
    attribution: pd.Series  # Return attribution by factor
    r_squared: float  # Model explanatory power
    alpha: float  # Unexplained excess return
    residuals: pd.Series  # Unexplained returns
    factor_correlations: pd.DataFrame  # Correlation matrix


class FactorAnalyzer:
    """
    Analyze portfolio returns using factor models
    
    Supports:
    - Fama-French 3-factor model
    - Carhart 4-factor model (adds momentum)
    - Fama-French 5-factor model (adds profitability & investment)
    - Custom factor models
    """
    
    def __init__(self, strategy_returns: pd.Series):
        """
        Initialize analyzer with strategy returns
        
        Args:
            strategy_returns: Daily returns of your strategy
        """
        self.strategy_returns = strategy_returns.dropna()
    
    def fetch_factor_data(self, 
                         factors: List[str] = ['Mkt-RF', 'SMB', 'HML', 'RF'],
                         start_date: str = None,
                         end_date: str = None) -> pd.DataFrame:
        """
        Fetch factor returns (simulated for now - in production use Kenneth French library)
        
        Common factors:
        - Mkt-RF: Market excess return
        - SMB: Small Minus Big (size factor)
        - HML: High Minus Low (value factor)
        - Mom: Momentum factor
        - RMW: Robust Minus Weak (profitability)
        - CMA: Conservative Minus Aggressive (investment)
        - RF: Risk-free rate
        """
        # In production, fetch from Kenneth French Data Library
        # For now, simulate factors based on typical characteristics
        
        np.random.seed(42)
        dates = self.strategy_returns.index
        
        # Simulated factor returns with realistic correlations
        n = len(dates)
        
        factor_data = pd.DataFrame(index=dates)
        
        if 'Mkt-RF' in factors:
            # Market factor: higher mean, moderate vol
            factor_data['Mkt-RF'] = np.random.normal(0.0004, 0.01, n)
        
        if 'SMB' in factors:
            # Size factor: small cap premium
            factor_data['SMB'] = np.random.normal(0.0001, 0.005, n)
        
        if 'HML' in factors:
            # Value factor: value premium
            factor_data['HML'] = np.random.normal(0.0001, 0.005, n)
        
        if 'Mom' in factors:
            # Momentum factor: autocorrelated
            mom = np.random.normal(0.0002, 0.008, n)
            # Add autocorrelation
            for i in range(1, n):
                mom[i] = 0.3 * mom[i-1] + 0.7 * mom[i]
            factor_data['Mom'] = mom
        
        if 'RMW' in factors:
            # Profitability factor
            factor_data['RMW'] = np.random.normal(0.00005, 0.003, n)
        
        if 'CMA' in factors:
            # Investment factor
            factor_data['CMA'] = np.random.normal(0.00005, 0.003, n)
        
        if 'RF' in factors:
            # Risk-free rate
            factor_data['RF'] = np.full(n, 0.00001)  # ~2.5% annual
        
        return factor_data
    
    def run_factor_regression(self, 
                             factor_data: pd.DataFrame,
                             excess_returns: bool = True) -> FactorAnalysisResults:
        """
        Run factor regression to decompose returns
        
        Model: R_portfolio = alpha + beta_1*F_1 + beta_2*F_2 + ... + epsilon
        
        Args:
            factor_data: DataFrame with factor returns
            excess_returns: Whether to use excess returns (subtract RF)
        """
        # Align dates
        common_dates = self.strategy_returns.index.intersection(factor_data.index)
        y = self.strategy_returns.loc[common_dates].values
        
        # Calculate excess returns if requested
        if excess_returns and 'RF' in factor_data.columns:
            rf = factor_data.loc[common_dates, 'RF'].values
            y = y - rf
            X_factors = [col for col in factor_data.columns if col != 'RF']
        else:
            X_factors = factor_data.columns.tolist()
        
        X = factor_data.loc[common_dates, X_factors].values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        betas = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        alpha = betas[0]
        factor_loadings = pd.Series(betas[1:], index=X_factors)
        
        # Calculate fitted values and residuals
        fitted = X_with_intercept @ betas
        residuals = y - fitted
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Attribution: how much each factor contributed
        attribution = pd.Series(index=['Alpha'] + X_factors)
        attribution['Alpha'] = alpha * len(y)  # Cumulative alpha
        
        for i, factor in enumerate(X_factors):
            factor_contribution = factor_loadings[factor] * factor_data.loc[common_dates, factor].sum()
            attribution[factor] = factor_contribution
        
        # Factor correlations
        factor_corr = factor_data[X_factors].corr()
        
        results = FactorAnalysisResults(
            factor_loadings=factor_loadings,
            factor_returns=factor_data.loc[common_dates],
            attribution=attribution,
            r_squared=r_squared,
            alpha=alpha * 252,  # Annualized
            residuals=pd.Series(residuals, index=common_dates),
            factor_correlations=factor_corr
        )
        
        return results
    
    def fama_french_3factor(self) -> FactorAnalysisResults:
        """Run Fama-French 3-factor model (Market, Size, Value)"""
        factors = self.fetch_factor_data(['Mkt-RF', 'SMB', 'HML', 'RF'])
        return self.run_factor_regression(factors)
    
    def carhart_4factor(self) -> FactorAnalysisResults:
        """Run Carhart 4-factor model (FF3 + Momentum)"""
        factors = self.fetch_factor_data(['Mkt-RF', 'SMB', 'HML', 'Mom', 'RF'])
        return self.run_factor_regression(factors)
    
    def fama_french_5factor(self) -> FactorAnalysisResults:
        """Run Fama-French 5-factor model (FF3 + Profitability + Investment)"""
        factors = self.fetch_factor_data(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
        return self.run_factor_regression(factors)
    
    def rolling_factor_loadings(self,
                                factor_data: pd.DataFrame,
                                window: int = 252) -> pd.DataFrame:
        """
        Calculate time-varying factor exposures
        
        Args:
            factor_data: Factor returns
            window: Rolling window size (days)
        
        Returns:
            DataFrame with rolling beta estimates
        """
        common_dates = self.strategy_returns.index.intersection(factor_data.index)
        y = self.strategy_returns.loc[common_dates]
        
        X_factors = [col for col in factor_data.columns if col != 'RF']
        X = factor_data.loc[common_dates, X_factors]
        
        # Calculate excess returns
        if 'RF' in factor_data.columns:
            y = y - factor_data.loc[common_dates, 'RF']
        
        rolling_betas = pd.DataFrame(index=common_dates, columns=['Alpha'] + X_factors)
        
        for i in range(window, len(common_dates)):
            window_dates = common_dates[i-window:i]
            y_window = y.loc[window_dates].values
            X_window = X.loc[window_dates].values
            
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X_window)), X_window])
            
            # Regression
            betas = np.linalg.lstsq(X_with_intercept, y_window, rcond=None)[0]
            
            rolling_betas.loc[common_dates[i], 'Alpha'] = betas[0]
            rolling_betas.loc[common_dates[i], X_factors] = betas[1:]
        
        return rolling_betas.astype(float)
    
    def factor_contribution_analysis(self, 
                                    results: FactorAnalysisResults) -> pd.DataFrame:
        """
        Detailed breakdown of factor contributions to returns
        """
        total_return = self.strategy_returns.sum()
        
        contributions = []
        for factor in results.attribution.index:
            contrib = results.attribution[factor]
            pct_of_total = (contrib / total_return) * 100 if total_return != 0 else 0
            
            contributions.append({
                'Factor': factor,
                'Contribution': contrib,
                'Pct of Total Return': pct_of_total,
                'Annualized': contrib * (252 / len(self.strategy_returns))
            })
        
        return pd.DataFrame(contributions)
    
    def style_analysis(self) -> Dict[str, float]:
        """
        Determine investment style from factor exposures
        
        Returns style scores:
        - Value vs Growth (HML loading)
        - Small vs Large (SMB loading)
        - Momentum strength
        - Market sensitivity (beta)
        """
        results = self.carhart_4factor()
        
        style = {
            'Market Beta': results.factor_loadings.get('Mkt-RF', 0),
            'Size Tilt': results.factor_loadings.get('SMB', 0),
            'Value Tilt': results.factor_loadings.get('HML', 0),
            'Momentum Exposure': results.factor_loadings.get('Mom', 0),
            'R-Squared': results.r_squared,
            'Alpha (Annual)': results.alpha
        }
        
        # Interpret tilts
        style['Style'] = self._interpret_style(style)
        
        return style
    
    def _interpret_style(self, style: Dict) -> str:
        """Interpret factor loadings into investment style"""
        descriptions = []
        
        # Market
        beta = style['Market Beta']
        if beta > 1.2:
            descriptions.append("High Beta (Aggressive)")
        elif beta < 0.8:
            descriptions.append("Low Beta (Defensive)")
        else:
            descriptions.append("Market-like Beta")
        
        # Size
        smb = style['Size Tilt']
        if smb > 0.3:
            descriptions.append("Small Cap Tilt")
        elif smb < -0.3:
            descriptions.append("Large Cap Tilt")
        
        # Value
        hml = style['Value Tilt']
        if hml > 0.3:
            descriptions.append("Value Tilt")
        elif hml < -0.3:
            descriptions.append("Growth Tilt")
        
        # Momentum
        mom = style['Momentum Exposure']
        if mom > 0.3:
            descriptions.append("Strong Momentum")
        
        return ", ".join(descriptions) if descriptions else "Market Neutral"


def print_factor_analysis_summary(results: FactorAnalysisResults, model_name: str = ""):
    """Pretty print factor analysis results"""
    
    print("=" * 70)
    if model_name:
        print(f"FACTOR ANALYSIS RESULTS - {model_name}")
    else:
        print("FACTOR ANALYSIS RESULTS")
    print("=" * 70)
    print()
    
    print("MODEL FIT:")
    print(f"  R-Squared: {results.r_squared:.4f} ({results.r_squared*100:.2f}% explained)")
    print(f"  Alpha (Annual): {results.alpha:.4f} ({results.alpha*100:.2f}%)")
    print()
    
    print("FACTOR LOADINGS (Betas):")
    for factor, loading in results.factor_loadings.items():
        significance = "***" if abs(loading) > 0.5 else "**" if abs(loading) > 0.3 else "*" if abs(loading) > 0.1 else ""
        print(f"  {factor:12s}: {loading:7.4f} {significance}")
    print()
    print("  * Significance: *** Strong, ** Moderate, * Weak")
    print()
    
    print("RETURN ATTRIBUTION:")
    total_attr = results.attribution.sum()
    for factor, contrib in results.attribution.items():
        pct = (contrib / total_attr * 100) if total_attr != 0 else 0
        print(f"  {factor:12s}: {contrib:8.4f} ({pct:5.1f}%)")
    print()
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
    
    # Run factor analysis
    analyzer = FactorAnalyzer(returns)
    results = analyzer.carhart_4factor()
    
    print_factor_analysis_summary(results, "Carhart 4-Factor Model")
    
    # Style analysis
    style = analyzer.style_analysis()
    print("\nINVESTMENT STYLE:")
    print(f"  {style['Style']}")
