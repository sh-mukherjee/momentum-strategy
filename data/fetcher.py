"""
data/fetcher.py - Data acquisition module (Updated for yfinance changes)
"""
import pandas as pd
import yfinance as yf
from typing import List
import warnings
warnings.filterwarnings('ignore')

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
            DataFrame with adjusted close prices (or close prices if adj close unavailable)
        """
        print(f"Fetching data for {len(tickers)} assets...")
        
        try:
            # Download data
            data = yf.download(
                tickers, 
                start=self.start_date, 
                end=self.end_date, 
                progress=False,
                auto_adjust=False  # Get both Close and Adj Close
            )
            
            # Handle single ticker vs multiple tickers
            if len(tickers) == 1:
                # Single ticker returns a simple DataFrame
                if 'Adj Close' in data.columns:
                    prices = data[['Adj Close']].copy()
                    prices.columns = tickers
                elif 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = tickers
                else:
                    raise ValueError(f"No price data found for {tickers[0]}")
            else:
                # Multiple tickers returns MultiIndex columns
                # Try Adj Close first, fallback to Close
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close'].copy()
                elif 'Close' in data.columns.get_level_values(0):
                    prices = data['Close'].copy()
                else:
                    raise ValueError("No price data found in download")
            
            # Ensure it's a DataFrame
            if not isinstance(prices, pd.DataFrame):
                prices = pd.DataFrame(prices)
            
            # Handle missing data
            # Forward fill first (for minor gaps)
            prices = prices.fillna(method='ffill')
            
            # Drop columns with too much missing data (>50% missing)
            missing_pct = prices.isna().sum() / len(prices)
            valid_columns = missing_pct[missing_pct < 0.5].index
            prices = prices[valid_columns]
            
            # Drop any remaining NaN rows
            prices = prices.dropna()
            
            if prices.empty:
                raise ValueError("No valid data after cleaning")
            
            # Report any tickers that were dropped
            dropped_tickers = set(tickers) - set(prices.columns)
            if dropped_tickers:
                print(f"⚠️  Warning: Dropped tickers due to insufficient data: {dropped_tickers}")
            
            print(f"✅ Data shape: {prices.shape}")
            print(f"✅ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
            print(f"✅ Assets: {list(prices.columns)}")
            
            return prices
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            
            # Fallback: Try fetching with auto_adjust=True
            print("🔄 Attempting fallback method...")
            try:
                data = yf.download(
                    tickers,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True  # This returns adjusted close as 'Close'
                )
                
                if len(tickers) == 1:
                    prices = data[['Close']].copy()
                    prices.columns = tickers
                else:
                    if 'Close' in data.columns.get_level_values(0):
                        prices = data['Close'].copy()
                    else:
                        prices = data.copy()
                
                prices = prices.fillna(method='ffill').dropna()
                
                if prices.empty:
                    raise ValueError("No valid data in fallback method")
                
                print(f"✅ Fallback successful! Data shape: {prices.shape}")
                return prices
                
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {str(fallback_error)}")
                raise
    
    def fetch_data_single_ticker(self, ticker: str) -> pd.Series:
        """
        Fetch data for a single ticker, returns a Series
        
        Args:
            ticker: Single ticker symbol
            
        Returns:
            Series with prices
        """
        df = self.fetch_data([ticker])
        return df[ticker]
    
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """
        Check which tickers are valid before fetching full history
        
        Args:
            tickers: List of ticker symbols to validate
            
        Returns:
            List of valid tickers
        """
        print(f"Validating {len(tickers)} tickers...")
        valid_tickers = []
        
        for ticker in tickers:
            try:
                # Try to fetch just 5 days of data to validate
                test_data = yf.download(
                    ticker,
                    period='5d',
                    progress=False
                )
                
                if not test_data.empty:
                    valid_tickers.append(ticker)
                else:
                    print(f"⚠️  Invalid ticker: {ticker}")
                    
            except Exception as e:
                print(f"⚠️  Error with {ticker}: {str(e)}")
        
        print(f"✅ Valid tickers: {len(valid_tickers)}/{len(tickers)}")
        return valid_tickers
    
    def get_available_columns(self, ticker: str) -> List[str]:
        """
        Check what columns are available for a ticker
        Useful for debugging
        """
        try:
            data = yf.download(ticker, period='5d', progress=False)
            print(f"Available columns for {ticker}:")
            print(data.columns.tolist())
            return data.columns.tolist()
        except Exception as e:
            print(f"Error checking columns: {e}")
            return []


# Helper function for quick testing
def test_data_fetcher():
    """Test the data fetcher with common tickers"""
    print("=" * 70)
    print("TESTING DATA FETCHER")
    print("=" * 70)
    
    fetcher = DataFetcher('2023-01-01', '2024-01-01')
    
    # Test with common tickers
    test_tickers = ['SPY', 'QQQ', 'AAPL']
    
    print("\n1. Testing fetch_data with multiple tickers...")
    try:
        data = fetcher.fetch_data(test_tickers)
        print(f"✅ Success! Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nFirst few rows:")
        print(data.head())
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n2. Testing single ticker...")
    try:
        single = fetcher.fetch_data_single_ticker('SPY')
        print(f"✅ Success! Shape: {single.shape}")
        print(f"\nFirst few values:")
        print(single.head())
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n3. Checking available columns...")
    fetcher.get_available_columns('SPY')
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_data_fetcher()
