"""Investing.com data adapter implementation.

This module provides an adapter for Investing.com data using web scraping techniques,
implementing the FinancialDataAdapter interface.
"""

import re
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from .base_adapter import FinancialDataAdapter
from .config import get_adapter_config
from .utils import HTTPClient, RateLimiter, normalize_symbol, validate_data_quality, retry_with_backoff
from stock_analysis.utils.exceptions import DataRetrievalError, NetworkError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class InvestingAdapter(FinancialDataAdapter):
    """Adapter for Investing.com data using web scraping.
    
    This adapter retrieves financial data from Investing.com through web scraping
    with proper rate limiting, session management, and error handling.
    """
    
    def __init__(self):
        """Initialize Investing.com adapter."""
        config = get_adapter_config('investing')
        super().__init__('investing', config.settings if config else {})
        
        # Initialize HTTP client with rate limiting
        rate_limit = config.rate_limit if config else 0.5
        timeout = config.timeout if config else 45
        max_retries = config.max_retries if config else 3
        
        self.http_client = HTTPClient(
            base_url="https://www.investing.com",
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            timeout=timeout,
            rate_limiter=RateLimiter(rate_limit),
            max_retries=max_retries
        )
        
        # Cache for symbol lookups and data
        self._symbol_cache = {}
        self._data_cache = {}
        self._cache_ttl = config.get_setting('cache_ttl', 600) if config else 600
        
        # Session management
        self._session_initialized = False
        self._last_request_time = 0
    
    def _initialize_session(self) -> None:
        """Initialize session with Investing.com."""
        if self._session_initialized:
            return
        
        try:
            # Make initial request to establish session
            response = self.http_client.get('/')
            self._session_initialized = True
            self.logger.info("Investing.com session initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Investing.com session: {str(e)}")
            raise DataRetrievalError(f"Failed to initialize session: {str(e)}")
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if still valid."""
        if cache_key in self._data_cache:
            cached_data, timestamp = self._data_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
            else:
                # Remove expired cache entry
                del self._data_cache[cache_key]
        return None
    
    def _set_cached_data(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache data with timestamp."""
        self._data_cache[cache_key] = (data, time.time())
    
    @retry_with_backoff(max_retries=3, exceptions=(NetworkError, requests.RequestException))
    def _search_symbol(self, symbol: str) -> Optional[Dict[str, str]]:
        """Search for symbol on Investing.com and return URL and details."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        
        try:
            self._initialize_session()
            
            # Search for the symbol
            search_url = f"/search/?q={symbol}"
            response = self.http_client.get(search_url)
            
            if 'content' in response:
                soup = BeautifulSoup(response['content'], 'html.parser')
                
                # Look for search results
                search_results = soup.find_all('a', class_='js-inner-all-results-quote-item')
                
                for result in search_results:
                    result_text = result.get_text().strip()
                    href = result.get('href', '')
                    
                    # Check if this matches our symbol
                    if symbol.upper() in result_text.upper():
                        symbol_info = {
                            'url': href,
                            'name': result_text,
                            'type': self._determine_security_type(href)
                        }
                        self._symbol_cache[symbol] = symbol_info
                        return symbol_info
                
                # If no exact match, try the first result
                if search_results:
                    first_result = search_results[0]
                    symbol_info = {
                        'url': first_result.get('href', ''),
                        'name': first_result.get_text().strip(),
                        'type': self._determine_security_type(first_result.get('href', ''))
                    }
                    self._symbol_cache[symbol] = symbol_info
                    return symbol_info
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching for symbol {symbol}: {str(e)}")
            return None
    
    def _determine_security_type(self, url: str) -> str:
        """Determine security type from URL."""
        if '/equities/' in url:
            return 'stock'
        elif '/etfs/' in url:
            return 'etf'
        elif '/indices/' in url:
            return 'index'
        elif '/commodities/' in url:
            return 'commodity'
        elif '/currencies/' in url:
            return 'currency'
        else:
            return 'unknown'
    
    def get_security_info(self, symbol: str) -> Dict[str, Any]:
        """Retrieve basic security information."""
        try:
            symbol = normalize_symbol(symbol)
            cache_key = f"security_info_{symbol}"
            
            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            self.logger.info(f"Retrieving security info for {symbol}")
            
            # Search for symbol
            symbol_info = self._search_symbol(symbol)
            if not symbol_info:
                raise DataRetrievalError(f"Symbol {symbol} not found on Investing.com")
            
            # Get detailed information from the symbol's page
            response = self.http_client.get(symbol_info['url'])
            
            if 'content' not in response:
                raise DataRetrievalError(f"Failed to retrieve page content for {symbol}")
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            
            # Extract basic information
            result = {
                'symbol': symbol,
                'name': symbol_info['name'],
                'type': symbol_info['type'],
                'url': symbol_info['url']
            }
            
            # Extract current price
            price_element = soup.find('span', {'data-test': 'instrument-price-last'})
            if not price_element:
                price_element = soup.find('span', class_='text-2xl')
                if not price_element:
                    price_element = soup.find('span', class_='last-price-value')
            
            if price_element:
                price_text = price_element.get_text().strip()
                price_clean = re.sub(r'[^\d.-]', '', price_text)
                try:
                    result['current_price'] = float(price_clean)
                except ValueError:
                    self.logger.warning(f"Could not parse price: {price_text}")
                    result['current_price'] = 0.0
            else:
                result['current_price'] = 0.0
            
            # Extract additional information from overview table
            overview_data = self._extract_overview_data(soup)
            result.update(overview_data)
            
            # Validate and cache result
            required_fields = ['symbol', 'name', 'current_price']
            validation = validate_data_quality(result, required_fields)
            
            if not validation['is_valid']:
                self.logger.warning(f"Data quality issues for {symbol}: {validation}")
            
            self._set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving security info for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve security info for {symbol}: {str(e)}")
    
    def _extract_overview_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract overview data from the page."""
        overview_data = {}
        
        try:
            # Look for overview table or data points
            overview_table = soup.find('div', class_='overview-data-table')
            if not overview_table:
                overview_table = soup.find('div', class_='instrument-metadata')
            
            if overview_table:
                # Extract key-value pairs
                rows = overview_table.find_all('div', class_='flex')
                for row in rows:
                    cells = row.find_all('div')
                    if len(cells) >= 2:
                        key = cells[0].get_text().strip().lower().replace(' ', '_')
                        value_text = cells[1].get_text().strip()
                        
                        # Parse numeric values
                        if key in ['market_cap', 'volume', 'avg_volume', 'shares_outstanding']:
                            overview_data[key] = self._parse_numeric_value(value_text)
                        elif key in ['pe_ratio', 'pb_ratio', 'dividend_yield', 'beta']:
                            overview_data[key] = self._parse_float_value(value_text)
                        else:
                            overview_data[key] = value_text
            
            # Extract sector and industry if available
            breadcrumb = soup.find('nav', class_='breadcrumb')
            if breadcrumb:
                links = breadcrumb.find_all('a')
                if len(links) >= 2:
                    overview_data['sector'] = links[-2].get_text().strip()
                if len(links) >= 3:
                    overview_data['industry'] = links[-3].get_text().strip()
        
        except Exception as e:
            self.logger.warning(f"Error extracting overview data: {str(e)}")
        
        return overview_data
    
    def _parse_numeric_value(self, value_text: str) -> Optional[float]:
        """Parse numeric value with suffixes (K, M, B, T)."""
        try:
            value_text = value_text.replace(',', '').strip()
            
            # Handle suffixes
            multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
            
            for suffix, multiplier in multipliers.items():
                if value_text.upper().endswith(suffix):
                    number_part = value_text[:-1]
                    return float(number_part) * multiplier
            
            # Try to parse as regular number
            return float(re.sub(r'[^\d.-]', '', value_text))
        
        except (ValueError, TypeError):
            return None
    
    def _parse_float_value(self, value_text: str) -> Optional[float]:
        """Parse float value, handling percentages."""
        try:
            value_text = value_text.replace(',', '').strip()
            
            # Handle percentage
            if value_text.endswith('%'):
                return float(value_text[:-1]) / 100
            
            # Try to parse as regular float
            return float(re.sub(r'[^\d.-]', '', value_text))
        
        except (ValueError, TypeError):
            return None
    
    def get_historical_prices(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Retrieve historical price data."""
        try:
            symbol = normalize_symbol(symbol)
            cache_key = f"historical_{symbol}_{period}_{interval}"
            
            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data and 'dataframe' in cached_data:
                return cached_data['dataframe']
            
            self.logger.info(f"Retrieving historical data for {symbol} (period={period}, interval={interval})")
            
            # Search for symbol
            symbol_info = self._search_symbol(symbol)
            if not symbol_info:
                raise DataRetrievalError(f"Symbol {symbol} not found on Investing.com")
            
            # Build historical data URL
            historical_url = f"{symbol_info['url']}-historical-data"
            
            # Get historical data page
            response = self.http_client.get(historical_url)
            
            if 'content' not in response:
                raise DataRetrievalError(f"Failed to retrieve historical data page for {symbol}")
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            
            # Extract historical data table
            historical_data = self._extract_historical_data(soup)
            
            if historical_data.empty:
                raise DataRetrievalError(f"No historical data found for {symbol}")
            
            # Cache result
            cache_data = {'dataframe': historical_data}
            self._set_cached_data(cache_key, cache_data)
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving historical data for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve historical data for {symbol}: {str(e)}")
    
    def _calculate_start_date(self, period: str, end_date: datetime) -> datetime:
        """Calculate start date based on period."""
        period_map = {
            '1d': timedelta(days=1),
            '5d': timedelta(days=5),
            '1mo': timedelta(days=30),
            '3mo': timedelta(days=90),
            '6mo': timedelta(days=180),
            '1y': timedelta(days=365),
            '2y': timedelta(days=730),
            '5y': timedelta(days=1825),
            '10y': timedelta(days=3650),
            'ytd': timedelta(days=(end_date - datetime(end_date.year, 1, 1)).days),
            'max': timedelta(days=7300)  # ~20 years
        }
        
        delta = period_map.get(period, timedelta(days=365))
        return end_date - delta
    
    def _extract_historical_data(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Extract historical data from the page."""
        try:
            # Look for historical data table
            table = soup.find('table', class_='historical-data-table')
            if not table:
                table = soup.find('table', {'data-test': 'historical-data-table'})
            
            if not table:
                return pd.DataFrame()
            
            # Extract headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                for th in header_row.find_all('th'):
                    headers.append(th.get_text().strip())
            
            # Extract data rows
            rows = []
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    row_data = []
                    for td in tr.find_all('td'):
                        row_data.append(td.get_text().strip())
                    if row_data:
                        rows.append(row_data)
            
            if not rows or not headers:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
            
            # Clean and convert data
            df = self._clean_historical_dataframe(df)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error extracting historical data: {str(e)}")
            return pd.DataFrame()
    
    def _clean_historical_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format historical data DataFrame."""
        try:
            # Rename columns to standard format
            column_mapping = {
                'Date': 'Date',
                'Price': 'Close',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Vol.': 'Volume',
                'Change %': 'Change_Pct'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.set_index('Date')
            
            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self._parse_numeric_value)
            
            # Sort by date
            df = df.sort_index()
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error cleaning historical DataFrame: {str(e)}")
            return df
    
    def get_financial_statements(self, symbol: str, statement_type: str = "income", period: str = "annual", years: int = 5) -> pd.DataFrame:
        """Retrieve financial statements."""
        try:
            symbol = normalize_symbol(symbol)
            cache_key = f"financials_{symbol}_{statement_type}_{period}"
            
            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data and 'dataframe' in cached_data:
                return cached_data['dataframe']
            
            self.logger.info(f"Retrieving {period} {statement_type} statement for {symbol}")
            
            # Search for symbol
            symbol_info = self._search_symbol(symbol)
            if not symbol_info:
                raise DataRetrievalError(f"Symbol {symbol} not found on Investing.com")
            
            # Build financials URL
            statement_map = {
                'income': 'income-statement',
                'balance': 'balance-sheet',
                'cash': 'cash-flow'
            }
            
            statement_url_part = statement_map.get(statement_type, 'income-statement')
            financials_url = f"{symbol_info['url']}-{statement_url_part}"
            
            # Get financial statements page
            response = self.http_client.get(financials_url)
            
            if 'content' not in response:
                raise DataRetrievalError(f"Failed to retrieve financial statements page for {symbol}")
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            
            # Extract financial data
            financial_data = self._extract_financial_statements(soup, period)
            
            if financial_data.empty:
                self.logger.warning(f"No {statement_type} statement data found for {symbol}")
                return pd.DataFrame()
            
            # Cache result
            cache_data = {'dataframe': financial_data}
            self._set_cached_data(cache_key, cache_data)
            
            return financial_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving {statement_type} statement for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve {statement_type} statement for {symbol}: {str(e)}")
    
    def _extract_financial_statements(self, soup: BeautifulSoup, period: str) -> pd.DataFrame:
        """Extract financial statements from the page."""
        try:
            # Look for financial data table
            table = soup.find('table', class_='genTbl')
            if not table:
                table = soup.find('table', {'data-test': 'financial-table'})
            
            if not table:
                return pd.DataFrame()
            
            # Extract data
            rows = []
            headers = []
            
            # Get headers from first row
            first_row = table.find('tr')
            if first_row:
                for th in first_row.find_all(['th', 'td']):
                    headers.append(th.get_text().strip())
            
            # Get data rows
            for tr in table.find_all('tr')[1:]:  # Skip header row
                row_data = []
                for td in tr.find_all(['td', 'th']):
                    row_data.append(td.get_text().strip())
                if row_data:
                    rows.append(row_data)
            
            if not rows or not headers:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers[:len(rows[0]) if rows else 0])
            
            # Clean the DataFrame
            df = self._clean_financial_dataframe(df)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error extracting financial statements: {str(e)}")
            return pd.DataFrame()
    
    def _clean_financial_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format financial statements DataFrame."""
        try:
            if df.empty:
                return df
            
            # Set first column as index (usually contains line items)
            if len(df.columns) > 0:
                df = df.set_index(df.columns[0])
            
            # Convert numeric columns
            for col in df.columns:
                if col != df.index.name:
                    df[col] = df[col].apply(self._parse_financial_value)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error cleaning financial DataFrame: {str(e)}")
            return df
    
    def _parse_financial_value(self, value_text: str) -> Optional[float]:
        """Parse financial values, handling negatives and suffixes."""
        try:
            if not value_text or value_text.strip() in ['-', 'N/A', '']:
                return None
            
            value_text = value_text.replace(',', '').strip()
            
            # Handle parentheses as negative
            is_negative = False
            if value_text.startswith('(') and value_text.endswith(')'):
                is_negative = True
                value_text = value_text[1:-1]
            
            # Handle suffixes
            multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
            
            for suffix, multiplier in multipliers.items():
                if value_text.upper().endswith(suffix):
                    number_part = value_text[:-1]
                    result = float(number_part) * multiplier
                    return -result if is_negative else result
            
            # Try to parse as regular number
            result = float(re.sub(r'[^\d.-]', '', value_text))
            return -result if is_negative else result
        
        except (ValueError, TypeError):
            return None
    
    def get_technical_indicators(self, symbol: str, indicators: List[str] = None) -> Dict[str, Any]:
        """Retrieve technical indicators."""
        try:
            symbol = normalize_symbol(symbol)
            
            if not indicators:
                indicators = ['rsi_14', 'macd', 'sma_20', 'sma_50', 'sma_200']
            
            cache_key = f"technicals_{symbol}_{','.join(indicators)}"
            
            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            self.logger.info(f"Retrieving technical indicators for {symbol}: {indicators}")
            
            # Search for symbol
            symbol_info = self._search_symbol(symbol)
            if not symbol_info:
                raise DataRetrievalError(f"Symbol {symbol} not found on Investing.com")
            
            # Get technical analysis page
            technical_url = f"{symbol_info['url']}-technical"
            response = self.http_client.get(technical_url)
            
            if 'content' not in response:
                self.logger.warning(f"Failed to retrieve technical analysis page for {symbol}")
                return {}
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            
            # Extract technical indicators
            result = self._extract_technical_indicators(soup, indicators)
            
            # Cache result
            self._set_cached_data(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving technical indicators for {symbol}: {str(e)}")
            return {}
    
    def _extract_technical_indicators(self, soup: BeautifulSoup, indicators: List[str]) -> Dict[str, Any]:
        """Extract technical indicators from the page."""
        result = {}
        
        try:
            # Look for technical indicators table or sections
            technical_section = soup.find('div', class_='technicalIndicatorsTbl')
            if not technical_section:
                technical_section = soup.find('div', {'data-test': 'technical-indicators'})
            
            if technical_section:
                # Extract indicator values
                for indicator in indicators:
                    value = self._find_indicator_value(technical_section, indicator)
                    if value is not None:
                        result[indicator] = value
            
            # If no specific technical section found, try to extract from overview
            if not result:
                overview_section = soup.find('div', class_='overview-data-table')
                if overview_section:
                    for indicator in indicators:
                        value = self._find_indicator_value(overview_section, indicator)
                        if value is not None:
                            result[indicator] = value
        
        except Exception as e:
            self.logger.warning(f"Error extracting technical indicators: {str(e)}")
        
        return result
    
    def _find_indicator_value(self, section: BeautifulSoup, indicator: str) -> Optional[float]:
        """Find specific indicator value in a section."""
        try:
            # Map indicator names to possible text patterns
            indicator_patterns = {
                'rsi_14': ['RSI', 'RSI (14)', 'Relative Strength Index'],
                'macd': ['MACD', 'MACD (12,26,9)'],
                'sma_20': ['SMA 20', 'Simple Moving Average 20', 'MA(20)'],
                'sma_50': ['SMA 50', 'Simple Moving Average 50', 'MA(50)'],
                'sma_200': ['SMA 200', 'Simple Moving Average 200', 'MA(200)']
            }
            
            patterns = indicator_patterns.get(indicator, [indicator])
            
            for pattern in patterns:
                # Look for the pattern in the section
                elements = section.find_all(text=re.compile(pattern, re.IGNORECASE))
                for element in elements:
                    # Find the parent element and look for the value
                    parent = element.parent
                    if parent:
                        # Look for numeric value in siblings or nearby elements
                        siblings = parent.find_next_siblings()
                        for sibling in siblings:
                            value_text = sibling.get_text().strip()
                            parsed_value = self._parse_float_value(value_text)
                            if parsed_value is not None:
                                return parsed_value
            
            return None
        
        except Exception as e:
            self.logger.debug(f"Error finding indicator {indicator}: {str(e)}")
            return None
    
    def get_market_data(self, data_type: str) -> Dict[str, Any]:
        """Retrieve market data like indices, commodities, forex."""
        try:
            cache_key = f"market_data_{data_type}"
            
            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            self.logger.info(f"Retrieving market data: {data_type}")
            
            if data_type == 'indices':
                result = self._get_market_indices()
            elif data_type == 'commodities':
                result = self._get_commodities()
            elif data_type == 'forex':
                result = self._get_forex()
            elif data_type == 'sectors':
                result = self._get_sector_performance()
            else:
                self.logger.warning(f"Unsupported market data type: {data_type}")
                return {}
            
            # Cache result
            self._set_cached_data(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data ({data_type}): {str(e)}")
            return {}
    
    def _get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices data."""
        try:
            response = self.http_client.get('/indices/major-indices')
            
            if 'content' not in response:
                return {}
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            return self._extract_market_data_table(soup, 'indices')
        
        except Exception as e:
            self.logger.error(f"Error getting market indices: {str(e)}")
            return {}
    
    def _get_commodities(self) -> Dict[str, Any]:
        """Get commodities data."""
        try:
            response = self.http_client.get('/commodities')
            
            if 'content' not in response:
                return {}
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            return self._extract_market_data_table(soup, 'commodities')
        
        except Exception as e:
            self.logger.error(f"Error getting commodities: {str(e)}")
            return {}
    
    def _get_forex(self) -> Dict[str, Any]:
        """Get forex data."""
        try:
            response = self.http_client.get('/currencies/major-currencies')
            
            if 'content' not in response:
                return {}
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            return self._extract_market_data_table(soup, 'forex')
        
        except Exception as e:
            self.logger.error(f"Error getting forex: {str(e)}")
            return {}    
        
        
    def _get_sector_performance(self) -> Dict[str, Any]:
        """Get sector performance data."""
        try:
            response = self.http_client.get('/equities/sectors')
            
            if 'content' not in response:
                return {}
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            return self._extract_market_data_table(soup, 'sectors')
        
        except Exception as e:
            self.logger.error(f"Error getting sector performance: {str(e)}")
            return {}
    
    def _extract_market_data_table(self, soup: BeautifulSoup, data_type: str) -> Dict[str, Any]:
        """Extract market data from table."""
        result = {}
        
        try:
            # Look for data table
            table = soup.find('table', class_='genTbl')
            if not table:
                table = soup.find('table', {'data-test': 'market-data-table'})
            
            if not table:
                return result
            
            # Extract rows
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    name = cells[0].get_text().strip()
                    price_text = cells[1].get_text().strip()
                    change_text = cells[2].get_text().strip() if len(cells) > 2 else ""
                    
                    price = self._parse_numeric_value(price_text)
                    change = self._parse_numeric_value(change_text)
                    
                    result[name] = {
                        'name': name,
                        'price': price,
                        'change': change,
                        'type': data_type
                    }
        
        except Exception as e:
            self.logger.error(f"Error extracting market data table: {str(e)}")
        
        return result
    
    def get_news(self, symbol: str = None, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve news articles."""
        try:
            cache_key = f"news_{symbol or 'general'}_{category or 'all'}_{limit}"
            
            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            self.logger.info(f"Retrieving news (symbol={symbol}, category={category}, limit={limit})")
            
            if symbol:
                # Get company-specific news
                result = self._get_company_news(symbol, limit)
            else:
                # Get general market news
                result = self._get_market_news(category, limit)
            
            # Cache result
            self._set_cached_data(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving news: {str(e)}")
            return []
    
    def _get_company_news(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Get company-specific news."""
        try:
            # Search for symbol
            symbol_info = self._search_symbol(symbol)
            if not symbol_info:
                return []
            
            # Get news page
            news_url = f"{symbol_info['url']}-news"
            response = self.http_client.get(news_url)
            
            if 'content' not in response:
                return []
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            return self._extract_news_articles(soup, limit)
        
        except Exception as e:
            self.logger.error(f"Error getting company news for {symbol}: {str(e)}")
            return []
    
    def _get_market_news(self, category: str, limit: int) -> List[Dict[str, Any]]:
        """Get general market news."""
        try:
            news_url = '/news'
            if category:
                news_url += f'/{category}'
            
            response = self.http_client.get(news_url)
            
            if 'content' not in response:
                return []
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            return self._extract_news_articles(soup, limit)
        
        except Exception as e:
            self.logger.error(f"Error getting market news: {str(e)}")
            return []
    
    def _extract_news_articles(self, soup: BeautifulSoup, limit: int) -> List[Dict[str, Any]]:
        """Extract news articles from the page."""
        articles = []
        
        try:
            # Look for news articles
            article_elements = soup.find_all('div', class_='largeTitle')
            if not article_elements:
                article_elements = soup.find_all('article')
            
            for element in article_elements[:limit]:
                article_data = self._extract_article_data(element)
                if article_data:
                    articles.append(article_data)
        
        except Exception as e:
            self.logger.error(f"Error extracting news articles: {str(e)}")
        
        return articles
    
    def _extract_article_data(self, element: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract data from a single news article element."""
        try:
            # Extract title
            title_element = element.find('a')
            if not title_element:
                return None
            
            title = title_element.get_text().strip()
            url = title_element.get('href', '')
            
            # Make URL absolute
            if url and not url.startswith('http'):
                url = f"https://www.investing.com{url}"
            
            # Extract publication date
            date_element = element.find('span', class_='date')
            published_at = None
            if date_element:
                date_text = date_element.get_text().strip()
                try:
                    published_at = datetime.strptime(date_text, '%b %d, %Y')
                except ValueError:
                    # Try alternative date formats
                    try:
                        published_at = datetime.strptime(date_text, '%m/%d/%Y')
                    except ValueError:
                        published_at = datetime.now()
            
            # Extract summary if available
            summary_element = element.find('p')
            summary = summary_element.get_text().strip() if summary_element else ""
            
            return {
                'title': title,
                'url': url,
                'published_at': published_at or datetime.now(),
                'summary': summary,
                'source': 'Investing.com'
            }
        
        except Exception as e:
            self.logger.debug(f"Error extracting article data: {str(e)}")
            return None
    
    def get_economic_data(self, indicator: str, region: str = None) -> pd.DataFrame:
        """Retrieve economic indicators data."""
        try:
            cache_key = f"economic_{indicator}_{region or 'global'}"
            
            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data and 'dataframe' in cached_data:
                return cached_data['dataframe']
            
            self.logger.info(f"Retrieving economic data: {indicator} (region={region})")
            
            # Get economic calendar page
            response = self.http_client.get('/economic-calendar')
            
            if 'content' not in response:
                return pd.DataFrame()
            
            soup = BeautifulSoup(response['content'], 'html.parser')
            
            # Extract economic data
            economic_data = self._extract_economic_data(soup, indicator, region)
            
            # Cache result
            if not economic_data.empty:
                cache_data = {'dataframe': economic_data}
                self._set_cached_data(cache_key, cache_data)
            
            return economic_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving economic data for {indicator}: {str(e)}")
            return pd.DataFrame()
    
    def _extract_economic_data(self, soup: BeautifulSoup, indicator: str, region: str) -> pd.DataFrame:
        """Extract economic data from the page."""
        try:
            # Look for economic calendar table
            table = soup.find('table', {'id': 'economicCalendarData'})
            if not table:
                table = soup.find('table', class_='genTbl')
            
            if not table:
                return pd.DataFrame()
            
            # Extract relevant rows
            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header
                cells = tr.find_all(['td', 'th'])
                if len(cells) >= 4:
                    event_name = cells[1].get_text().strip()
                    
                    # Check if this row matches our indicator
                    if indicator.lower() in event_name.lower():
                        row_data = {
                            'date': cells[0].get_text().strip(),
                            'event': event_name,
                            'actual': cells[2].get_text().strip(),
                            'forecast': cells[3].get_text().strip() if len(cells) > 3 else '',
                            'previous': cells[4].get_text().strip() if len(cells) > 4 else ''
                        }
                        rows.append(row_data)
            
            if not rows:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Clean and convert data
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            for col in ['actual', 'forecast', 'previous']:
                if col in df.columns:
                    df[col] = df[col].apply(self._parse_economic_value)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error extracting economic data: {str(e)}")
            return pd.DataFrame()
    
    def _parse_economic_value(self, value_text: str) -> Optional[float]:
        """Parse economic indicator values."""
        try:
            if not value_text or value_text.strip() in ['-', 'N/A', '']:
                return None
            
            # Remove percentage signs and other characters
            value_text = value_text.replace('%', '').replace(',', '').strip()
            
            # Try to parse as float
            return float(re.sub(r'[^\d.-]', '', value_text))
        
        except (ValueError, TypeError):
            return None