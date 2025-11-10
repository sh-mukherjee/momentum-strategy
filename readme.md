# 🚀 Multi-Asset Momentum Trading Strategy

A comprehensive quantitative trading system implementing momentum strategies across equities, FX, and futures with advanced risk analytics, Monte Carlo simulation, factor analysis, and transaction cost optimization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Strategy Dashboard](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Interactive+Strategy+Dashboard)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Strategy Overview](#-strategy-overview)
- [Analysis Tools](#-analysis-tools)
- [Screenshots](#-screenshots)
- [Configuration](#️-configuration)
- [Performance Metrics](#-performance-metrics)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## ✨ Features

### 🎯 Core Strategy
- **Time-Series Momentum**: Trade based on asset's own historical performance
- **Cross-Sectional Momentum**: Rank assets relative to each other
- **Multi-Asset Support**: Equities, FX pairs, and futures contracts
- **Risk Parity**: Volatility-based position sizing
- **Transaction Cost Modeling**: Realistic cost assumptions

### 📊 Advanced Analytics
- **📈 Monte Carlo Simulation**: 10,000+ scenario projections with VaR/CVaR
- **🧬 Factor Exposure Analysis**: Fama-French 3/4/5-factor models
- **💰 Transaction Cost Sensitivity**: Optimize rebalancing frequency
- **📉 Performance Attribution**: Understand return drivers
- **🎲 Risk Assessment**: Comprehensive drawdown and probability analysis

### 🎨 Interactive Visualization
- **Plotly Charts**: Fully interactive with zoom, pan, hover tooltips
- **Real-Time Updates**: Dynamic parameter adjustment
- **Multiple Views**: Cumulative returns, drawdowns, factor loadings, cost analysis
- **Export Ready**: Download results as CSV, charts as PNG/SVG

### 🌐 Dual Interface
- **Web Application**: Beautiful Streamlit dashboard
- **Command-Line**: Automated backtesting and batch processing

## 🚀 Quick Start

### Web Application (Recommended)

```bash
# Clone the repository
git clone https://github.com/sh-mukherjee/momentum-strategy.git
cd momentum-strategy

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start exploring!

### Command-Line Interface

```bash
# Run with defaults
python main.py

# Run with custom parameters
python main.py --start-date 2020-01-01 --target-vol 0.20 --tickers SPY QQQ AAPL
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/momentum-strategy.git
cd momentum-strategy
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import streamlit; import plotly; print('✅ All dependencies installed!')"
```

## 📁 Project Structure

```
momentum-strategy/
│
├── data/                      # Data acquisition
│   ├── __init__.py
│   └── fetcher.py            # Market data fetching
│
├── signals/                   # Signal generation
│   ├── __init__.py
│   └── momentum.py           # Momentum signals
│
├── risk/                      # Risk management
│   ├── __init__.py
│   └── manager.py            # Position sizing
│
├── backtest/                  # Backtesting engine
│   ├── __init__.py
│   └── engine.py             # Performance calculation
│
├── simulation/                # Monte Carlo simulation
│   ├── __init__.py
│   └── monte_carlo.py        # Risk projections
│
├── analysis/                  # Advanced analytics
│   ├── __init__.py
│   ├── factor_exposure.py    # Factor analysis
│   └── transaction_costs.py  # Cost analysis
│
├── visualization/             # Interactive charts
│   ├── __init__.py
│   └── plots.py              # Plotly visualizations
│
├── app.py                     # Streamlit web app
├── main.py                    # Command-line interface
├── config.py                  # Default configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 💻 Usage

### Web Application

#### 1. Configure Strategy Parameters

**Date Range & Assets**
- Select start and end dates
- Choose from preset asset groups (Equities, FX, Futures)
- Add custom tickers

**Momentum Parameters**
- Short-term lookback (default: 20 days)
- Medium-term lookback (default: 60 days)
- Long-term lookback (default: 120 days)
- Cross-sectional lookback (default: 60 days)

**Risk Management**
- Target volatility (default: 15%)
- Volatility lookback window (default: 60 days)
- Transaction cost (default: 10 bps)

#### 2. Enable Advanced Analysis

**Monte Carlo Simulation**
- Choose method: Historical Bootstrap, Parametric, or GBM
- Set number of simulations (1,000 - 50,000)
- Project forward (30 - 504 days)

**Factor Analysis**
- Select factor model (FF3, Carhart 4, FF5)
- View factor loadings and attribution
- Analyze rolling exposures

**Transaction Cost Analysis**
- Sensitivity across cost levels
- Optimal rebalancing frequency
- Cost breakdown by asset

#### 3. Run Backtest

Click **"🚀 Run Backtest"** and view:
- Performance metrics
- Interactive charts
- Risk analytics
- Factor exposures
- Cost analysis

### Command-Line Interface

```bash
# Basic usage with defaults
python main.py

# Custom date range
python main.py --start-date 2020-01-01 --end-date 2023-12-31

# Custom target volatility
python main.py --target-vol 0.20

# Custom asset universe
python main.py --tickers SPY QQQ IWM GLD TLT

# Combine parameters
python main.py \
  --start-date 2020-01-01 \
  --target-vol 0.18 \
  --tickers SPY QQQ EEM GLD
```

## 📈 Strategy Overview

### Time-Series Momentum

Identifies trending assets by looking at their own past performance:

```python
# Long if positive return over lookback period
# Short if negative return
signal = sign(price_today / price_N_days_ago - 1)
```

**Multiple timeframes** (20/60/120 days) are averaged for robustness.

### Cross-Sectional Momentum

Ranks assets relative to each other:

```python
# Long top 30% performers
# Short bottom 30% performers
# Neutral middle 40%
ranks = rank(returns)
signal = 1 if rank > 0.7 else -1 if rank < 0.3 else 0
```

### Combined Signal

```python
final_signal = (time_series_signal + cross_sectional_signal) / 2
```

### Risk Management

**Inverse Volatility Weighting** (Risk Parity):
```python
weight_i = (1 / volatility_i) / sum(1 / volatility_j)
```

Higher volatility assets get smaller positions, equalizing risk contribution.

**Target Volatility Scaling**:
```python
scaled_position = weight × (target_vol / portfolio_vol)
```

Dynamically adjusts exposure to maintain consistent risk.

## 🔬 Analysis Tools

### 1. Monte Carlo Simulation

**What it does**: Projects 10,000+ future scenarios to understand risk

**Key Metrics**:
- **VaR (Value at Risk)**: Maximum expected loss at 95%/99% confidence
- **CVaR (Conditional VaR)**: Average loss in worst-case scenarios
- **Probability Analysis**: Likelihood of profit/loss thresholds
- **Drawdown Projections**: Expected peak-to-trough declines

**Methods**:
- Historical Bootstrap (recommended)
- Parametric (assumes normal distribution)
- Geometric Brownian Motion (theoretical)

### 2. Factor Exposure Analysis

**What it does**: Decomposes returns into known risk factors

**Fama-French 3-Factor**:
- **Mkt-RF**: Market excess return
- **SMB**: Size premium (small cap - large cap)
- **HML**: Value premium (value - growth)

**Carhart 4-Factor** (adds):
- **Mom**: Momentum factor

**Fama-French 5-Factor** (adds):
- **RMW**: Profitability factor
- **CMA**: Investment factor

**Output**:
- Factor loadings (betas)
- Alpha (skill-based return)
- R² (model fit)
- Return attribution

### 3. Transaction Cost Sensitivity

**What it does**: Tests strategy viability under realistic trading costs

**Analysis Types**:
- **Sensitivity**: Performance at 0-50 bps
- **Turnover**: Daily, annual, cumulative metrics
- **Rebalancing**: Optimal frequency (daily/weekly/monthly)
- **Breakeven**: Maximum tolerable cost
- **By Asset**: Which assets are most expensive
- **Slippage**: Market impact modeling

## 📸 Screenshots

### Main Dashboard
![Dashboard](https://via.placeholder.com/800x450/2ca02c/ffffff?text=Strategy+Dashboard+with+Performance+Metrics)

### Monte Carlo Simulation
![Monte Carlo](https://via.placeholder.com/800x450/1f77b4/ffffff?text=10%2C000+Scenario+Projections)

### Factor Analysis
![Factor Analysis](https://via.placeholder.com/800x450/ff7f0e/ffffff?text=Factor+Loadings+and+Attribution)

### Transaction Costs
![Cost Analysis](https://via.placeholder.com/800x450/d62728/ffffff?text=Cost+Sensitivity+Dashboard)

## ⚙️ Configuration

### config.py

Default parameters for command-line usage:

```python
# Date range
DEFAULT_START_DATE = '2015-01-01'
DEFAULT_END_DATE = '2024-12-31'

# Asset universe
DEFAULT_EQUITIES = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ']
DEFAULT_FX_PAIRS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
DEFAULT_FUTURES = ['GC=F', 'CL=F', 'ZB=F']

# Strategy parameters
DEFAULT_MOMENTUM_LOOKBACKS = [20, 60, 120]
DEFAULT_TARGET_VOLATILITY = 0.15
DEFAULT_TRANSACTION_COST = 0.001  # 10 basis points
```

### Customization

**Add new asset classes**:
```python
# In config.py
DEFAULT_CRYPTO = ['BTC-USD', 'ETH-USD']
```

**Modify signal generation**:
```python
# In signals/momentum.py
def custom_signal(self, prices):
    # Your logic here
    return signals
```

**Change risk model**:
```python
# In risk/manager.py
def custom_risk_model(self, signals, prices):
    # Your logic here
    return positions
```

## 📊 Performance Metrics

The strategy calculates comprehensive performance metrics:

### Returns
- Total Return
- Annual Return
- Monthly Returns
- Rolling Returns

### Risk
- Annual Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Average Drawdown
- Calmar Ratio

### Monte Carlo
- VaR (90%, 95%, 99%)
- CVaR (90%, 95%, 99%)
- Probability of Profit
- Expected Shortfall

### Factor Analysis
- Alpha
- Factor Loadings (Betas)
- R-Squared
- Return Attribution

### Transaction Costs
- Total Turnover
- Annualized Turnover
- Transaction Costs Paid
- Cost as % of Return
- Breakeven Cost

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution

- [ ] Add more factor models (custom factors, macro factors)
- [ ] Implement machine learning signals
- [ ] Add live trading integration (Alpaca, Interactive Brokers)
- [ ] Enhance visualization (3D plots, animations)
- [ ] Add more asset classes (crypto, commodities)
- [ ] Improve documentation
- [ ] Add unit tests
- [ ] Performance optimization

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Trading involves risk of loss
- Test thoroughly before deploying real capital
- Consider transaction costs, slippage, and market impact
- Consult a financial advisor for investment decisions

## 📚 References

### Academic Papers
- Fama, E. F., & French, K. R. (1993). "Common risk factors in the returns on stocks and bonds"
- Carhart, M. M. (1997). "On Persistence in Mutual Fund Performance"
- Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers"
- Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). "Value and Momentum Everywhere"

### Data Sources
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [Yahoo Finance](https://finance.yahoo.com)
- [FRED Economic Data](https://fred.stlouisfed.org)

### Tools & Libraries
- [Streamlit](https://streamlit.io) - Web framework
- [Plotly](https://plotly.com) - Interactive visualizations
- [yfinance](https://github.com/ranaroussi/yfinance) - Market data
- [NumPy](https://numpy.org) - Numerical computing
- [Pandas](https://pandas.pydata.org) - Data manipulation

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/momentum-strategy](https://github.com/yourusername/momentum-strategy)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for quantitative traders

</div>
