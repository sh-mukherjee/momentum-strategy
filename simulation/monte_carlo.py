"""
simulation/monte_carlo.py - Monte Carlo simulation for risk assessment
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results"""
    simulated_paths: np.ndarray  # Shape: (n_simulations, n_days)
    percentiles: Dict[int, np.ndarray]  # Percentile paths
    final_values: np.ndarray  # Final portfolio values
    metrics: Dict[str, float]  # Risk metrics
    var_cvar: Dict[str, float]  # Value at Risk metrics


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio risk assessment
    
    Supports multiple simulation methods:
    1. Historical bootstrap (resampling actual returns)
    2. Parametric (assumes normal distribution)
    3. Geometric Brownian Motion (GBM)
    """
    
    def __init__(self, returns: pd.Series, initial_value: float = 1.0):
        """
        Initialize simulator with historical returns
        
        Args:
            returns: Historical daily returns series
            initial_value: Starting portfolio value (default: 1.0 = 100%)
        """
        self.returns = returns.dropna()
        self.initial_value = initial_value
        
        # Calculate statistics from historical returns
        self.mean_return = self.returns.mean()
        self.std_return = self.returns.std()
        self.skewness = self.returns.skew()
        self.kurtosis = self.returns.kurtosis()
    
    def simulate_historical_bootstrap(self, 
                                      n_simulations: int = 10000,
                                      n_days: int = 252,
                                      random_seed: int = None) -> MonteCarloResults:
        """
        Bootstrap simulation using historical returns
        Randomly samples from actual historical returns with replacement
        
        Args:
            n_simulations: Number of simulation paths
            n_days: Number of days to simulate forward
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Randomly sample returns with replacement
        simulated_returns = np.random.choice(
            self.returns.values,
            size=(n_simulations, n_days),
            replace=True
        )
        
        # Calculate cumulative paths
        simulated_paths = self.initial_value * np.cumprod(1 + simulated_returns, axis=1)
        
        return self._calculate_results(simulated_paths, n_days)
    
    def simulate_parametric(self,
                           n_simulations: int = 10000,
                           n_days: int = 252,
                           random_seed: int = None) -> MonteCarloResults:
        """
        Parametric simulation assuming normal distribution
        Uses mean and std from historical returns
        
        Args:
            n_simulations: Number of simulation paths
            n_days: Number of days to simulate forward
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random returns from normal distribution
        simulated_returns = np.random.normal(
            self.mean_return,
            self.std_return,
            size=(n_simulations, n_days)
        )
        
        # Calculate cumulative paths
        simulated_paths = self.initial_value * np.cumprod(1 + simulated_returns, axis=1)
        
        return self._calculate_results(simulated_paths, n_days)
    
    def simulate_gbm(self,
                     n_simulations: int = 10000,
                     n_days: int = 252,
                     random_seed: int = None) -> MonteCarloResults:
        """
        Geometric Brownian Motion simulation
        Classic financial modeling approach
        
        dS/S = μ*dt + σ*dW
        
        Args:
            n_simulations: Number of simulation paths
            n_days: Number of days to simulate forward
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Annualized parameters
        mu = self.mean_return * 252
        sigma = self.std_return * np.sqrt(252)
        dt = 1/252  # Daily time step
        
        # Generate random walks
        dW = np.random.normal(0, np.sqrt(dt), size=(n_simulations, n_days))
        
        # GBM formula
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * dW
        
        # Calculate cumulative returns
        returns = drift + diffusion
        simulated_paths = self.initial_value * np.exp(np.cumsum(returns, axis=1))
        
        return self._calculate_results(simulated_paths, n_days)
    
    def _calculate_results(self, 
                          simulated_paths: np.ndarray,
                          n_days: int) -> MonteCarloResults:
        """Calculate comprehensive results from simulated paths"""
        
        # Calculate percentiles
        percentiles = {
            5: np.percentile(simulated_paths, 5, axis=0),
            25: np.percentile(simulated_paths, 25, axis=0),
            50: np.percentile(simulated_paths, 50, axis=0),
            75: np.percentile(simulated_paths, 75, axis=0),
            95: np.percentile(simulated_paths, 95, axis=0)
        }
        
        # Final values distribution
        final_values = simulated_paths[:, -1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(simulated_paths, final_values, n_days)
        
        # Calculate VaR and CVaR
        var_cvar = self._calculate_var_cvar(final_values)
        
        return MonteCarloResults(
            simulated_paths=simulated_paths,
            percentiles=percentiles,
            final_values=final_values,
            metrics=metrics,
            var_cvar=var_cvar
        )
    
    def _calculate_metrics(self, 
                          paths: np.ndarray,
                          final_values: np.ndarray,
                          n_days: int) -> Dict[str, float]:
        """Calculate risk metrics from simulation results"""
        
        # Expected final value
        expected_value = np.mean(final_values)
        
        # Probability of profit (ending above initial value)
        prob_profit = np.mean(final_values > self.initial_value)
        
        # Probability of loss > 10%, 20%, 50%
        prob_loss_10 = np.mean(final_values < self.initial_value * 0.9)
        prob_loss_20 = np.mean(final_values < self.initial_value * 0.8)
        prob_loss_50 = np.mean(final_values < self.initial_value * 0.5)
        
        # Maximum drawdown across all paths
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        avg_max_drawdown = np.mean(np.min(drawdowns, axis=1))
        
        # Expected return (annualized)
        total_return = expected_value / self.initial_value - 1
        years = n_days / 252
        expected_annual_return = (1 + total_return) ** (1/years) - 1
        
        # Volatility of final values (annualized)
        volatility = np.std(final_values) / self.initial_value
        annual_volatility = volatility * np.sqrt(252/n_days)
        
        return {
            'expected_value': expected_value,
            'expected_return': total_return,
            'expected_annual_return': expected_annual_return,
            'annual_volatility': annual_volatility,
            'prob_profit': prob_profit,
            'prob_loss_10pct': prob_loss_10,
            'prob_loss_20pct': prob_loss_20,
            'prob_loss_50pct': prob_loss_50,
            'max_drawdown': max_drawdown,
            'avg_max_drawdown': avg_max_drawdown,
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values)
        }
    
    def _calculate_var_cvar(self, final_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        
        VaR: Maximum expected loss at a given confidence level
        CVaR: Average loss beyond VaR (tail risk)
        """
        
        # Calculate returns from initial value
        returns = (final_values / self.initial_value) - 1
        
        var_cvar = {}
        
        for confidence in [0.90, 0.95, 0.99]:
            # VaR: Loss at the confidence percentile
            var = -np.percentile(returns, (1 - confidence) * 100)
            
            # CVaR: Average of losses worse than VaR
            cvar = -np.mean(returns[returns < -var])
            
            var_cvar[f'VaR_{int(confidence*100)}'] = var
            var_cvar[f'CVaR_{int(confidence*100)}'] = cvar
        
        return var_cvar
    
    def compare_methods(self,
                       n_simulations: int = 10000,
                       n_days: int = 252,
                       random_seed: int = 42) -> Dict[str, MonteCarloResults]:
        """
        Compare all three simulation methods
        
        Returns:
            Dictionary with results for each method
        """
        results = {
            'Historical Bootstrap': self.simulate_historical_bootstrap(
                n_simulations, n_days, random_seed
            ),
            'Parametric (Normal)': self.simulate_parametric(
                n_simulations, n_days, random_seed
            ),
            'Geometric Brownian Motion': self.simulate_gbm(
                n_simulations, n_days, random_seed
            )
        }
        
        return results
    
    def stress_test(self,
                   scenarios: Dict[str, Tuple[float, float]],
                   n_simulations: int = 10000,
                   n_days: int = 252,
                   random_seed: int = None) -> Dict[str, MonteCarloResults]:
        """
        Run stress test scenarios with different return/volatility assumptions
        
        Args:
            scenarios: Dict of {name: (mean_return, std_return)}
            Example: {
                'Base Case': (0.0005, 0.015),
                'Bull Market': (0.001, 0.012),
                'Bear Market': (-0.0005, 0.025),
                '2008 Crisis': (-0.002, 0.035)
            }
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        results = {}
        
        for scenario_name, (mean_ret, std_ret) in scenarios.items():
            # Generate returns with scenario parameters
            simulated_returns = np.random.normal(
                mean_ret,
                std_ret,
                size=(n_simulations, n_days)
            )
            
            # Calculate paths
            simulated_paths = self.initial_value * np.cumprod(
                1 + simulated_returns, axis=1
            )
            
            results[scenario_name] = self._calculate_results(simulated_paths, n_days)
        
        return results
    
    def confidence_intervals(self,
                           results: MonteCarloResults,
                           n_days: int = 252) -> pd.DataFrame:
        """
        Calculate confidence intervals over time
        
        Returns DataFrame with columns for each percentile
        """
        time_index = pd.date_range(
            start=pd.Timestamp.now(),
            periods=n_days,
            freq='D'
        )
        
        ci_df = pd.DataFrame(index=time_index)
        
        for percentile, values in results.percentiles.items():
            ci_df[f'P{percentile}'] = values
        
        return ci_df


def print_simulation_summary(results: MonteCarloResults, method_name: str = ""):
    """Pretty print simulation results"""
    
    print("=" * 70)
    if method_name:
        print(f"MONTE CARLO SIMULATION RESULTS - {method_name}")
    else:
        print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 70)
    print()
    
    print("EXPECTED OUTCOMES:")
    print(f"  Expected Final Value: {results.metrics['expected_value']:.3f}")
    print(f"  Expected Return: {results.metrics['expected_return']:.2%}")
    print(f"  Expected Annual Return: {results.metrics['expected_annual_return']:.2%}")
    print(f"  Annual Volatility: {results.metrics['annual_volatility']:.2%}")
    print()
    
    print("PROBABILITY ANALYSIS:")
    print(f"  Probability of Profit: {results.metrics['prob_profit']:.2%}")
    print(f"  Probability of >10% Loss: {results.metrics['prob_loss_10pct']:.2%}")
    print(f"  Probability of >20% Loss: {results.metrics['prob_loss_20pct']:.2%}")
    print(f"  Probability of >50% Loss: {results.metrics['prob_loss_50pct']:.2%}")
    print()
    
    print("DRAWDOWN ANALYSIS:")
    print(f"  Worst Drawdown (Single Path): {results.metrics['max_drawdown']:.2%}")
    print(f"  Average Max Drawdown: {results.metrics['avg_max_drawdown']:.2%}")
    print()
    
    print("VALUE AT RISK (VaR) & CONDITIONAL VAR (CVaR):")
    print(f"  VaR (90%):  {results.var_cvar['VaR_90']:.2%}  |  CVaR (90%):  {results.var_cvar['CVaR_90']:.2%}")
    print(f"  VaR (95%):  {results.var_cvar['VaR_95']:.2%}  |  CVaR (95%):  {results.var_cvar['CVaR_95']:.2%}")
    print(f"  VaR (99%):  {results.var_cvar['VaR_99']:.2%}  |  CVaR (99%):  {results.var_cvar['CVaR_99']:.2%}")
    print()
    
    print("FINAL VALUE DISTRIBUTION:")
    print(f"  5th Percentile:  {results.percentiles[5][-1]:.3f}")
    print(f"  25th Percentile: {results.percentiles[25][-1]:.3f}")
    print(f"  50th Percentile: {results.percentiles[50][-1]:.3f}")
    print(f"  75th Percentile: {results.percentiles[75][-1]:.3f}")
    print(f"  95th Percentile: {results.percentiles[95][-1]:.3f}")
    print()
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Generate sample returns
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.0005, 0.015, 1000))
    
    # Initialize simulator
    simulator = MonteCarloSimulator(sample_returns)
    
    # Run simulation
    results = simulator.simulate_historical_bootstrap(
        n_simulations=10000,
        n_days=252
    )
    
    # Print results
    print_simulation_summary(results, "Historical Bootstrap")
