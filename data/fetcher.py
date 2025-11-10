"""
Data acquisition module
"""
import pandas as pd
import yfinance as yf
from typing import List

class DataFetcher:
    """Fetch and prepare multi-asset market data"""
    
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch historical data for given tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with adjusted close prices
        """
        print(f"Fetching data for {len(tickers)} assets...")
        data = yf.download(tickers, start=self.start_date, end=self.end_date, 
                          progress=False)['Adj Close']
        
        # Handle missing data
        data = data.fillna(method='ffill').dropna()
        
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        return data
