"""
Configuration settings for the momentum strategy
"""

# Date range
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# Asset universe
EQUITIES = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ']
FX_PAIRS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
FUTURES = ['GC=F', 'CL=F', 'ZB=F']

# Strategy parameters
MOMENTUM_LOOKBACKS = [20, 60, 120]
CROSS_SECTIONAL_LOOKBACK = 60

# Risk management
TARGET_VOLATILITY = 0.15
VOLATILITY_LOOKBACK = 60

# Trading costs
TRANSACTION_COST = 0.001  # 10 basis points

# Visualization
ROLLING_SHARPE_WINDOW = 252
