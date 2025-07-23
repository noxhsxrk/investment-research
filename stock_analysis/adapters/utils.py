"""Utility functions for data source adapters.

This module provides common utilities for HTTP requests, rate limiting,
error handling, and data processing used by adapter implementations.
"""

import time
import random
import requests
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import json

from stock_analysis.utils.exceptions import DataRetrievalError, NetworkError, RateLimitError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 5):
        """Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size for requests
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
    
    def acquire(self) -> None:
        """Acquire a token for making a request."""
        current_time = time.time()
        
        # Add tokens based on elapsed time
        elapsed = current_time - self.last_update
        self.tokens = min(self.burst_size, self.tokens + elapsed * self.requests_per_second)
        self.last_update = current_time
        
        if self.tokens < 1:
            # Need to wait
            wait_time = (1 - self.tokens) / self.requests_per_second
            logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
            self.tokens = 0
        else:
            self.tokens -= 1


class HTTPClient:
    """HTTP client with rate limiting, retry logic, and error handling."""
    
    def __init__(
        self, 
        base_url: str = None,
        headers: Dict[str, str] = None,
        timeout: int = 30,
        rate_limiter: RateLimiter = None,
        max_retries: int = 3
    ):
        """Initialize HTTP client.
        
        Args:
            base_url: Base URL for requests
            headers: Default headers for requests
            timeout: Request timeout in seconds
            rate_limiter: Rate limiter instance
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url or ""
        self.headers = headers or {}
        self.timeout = timeout
        self.rate_limiter = rate_limiter or RateLimiter()
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Full URL
        """
        if endpoint.startswith('http'):
            return endpoint
        return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle HTTP response and extract data.
        
        Args:
            response: HTTP response object
            
        Returns:
            Response data as dictionary
            
        Raises:
            DataRetrievalError: If response indicates an error
        """
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        
        if response.status_code >= 400:
            error_msg = f"HTTP {response.status_code}: {response.reason}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg += f" - {error_data['error']}"
                elif 'message' in error_data:
                    error_msg += f" - {error_data['message']}"
            except (json.JSONDecodeError, ValueError):
                pass
            
            raise DataRetrievalError(error_msg)
        
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, return text content
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            return {'content': response.text}
    
    def get(
        self, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make GET request with rate limiting and retry logic.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data
            
        Raises:
            DataRetrievalError: If request fails after all retries
        """
        return self._request('GET', endpoint, params=params, headers=headers, **kwargs)
    
    def post(
        self, 
        endpoint: str, 
        data: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make POST request with rate limiting and retry logic.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data
            
        Raises:
            DataRetrievalError: If request fails after all retries
        """
        return self._request('POST', endpoint, data=data, json=json_data, headers=headers, **kwargs)
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data
            
        Raises:
            DataRetrievalError: If request fails after all retries
        """
        url = self._build_url(endpoint)
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                self.rate_limiter.acquire()
                
                # Make request
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                
                return self._handle_response(response)
                
            except RateLimitError as e:
                # For rate limit errors, wait longer before retry
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
                    logger.warning(f"Rate limit hit, waiting {wait_time:.2f} seconds before retry")
                    time.sleep(wait_time)
                    last_exception = e
                else:
                    raise e
                    
            except (requests.RequestException, NetworkError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}. "
                                 f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed: {str(e)}")
        
        # If we get here, all attempts failed
        raise DataRetrievalError(f"Failed to retrieve data after {self.max_retries + 1} attempts: {str(last_exception)}")


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for backoff time
        max_backoff: Maximum backoff time in seconds
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = min(backoff_factor ** attempt, max_backoff)
                        wait_time += random.uniform(0, wait_time * 0.1)  # Add jitter
                        logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): "
                                     f"{str(e)}. Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {str(e)}")
            
            raise last_exception
        
        return wrapper
    return decorator


def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol format.
    
    Args:
        symbol: Stock symbol to normalize
        
    Returns:
        Normalized symbol
    """
    if not symbol:
        return symbol
    
    # Convert to uppercase and strip whitespace
    symbol = symbol.upper().strip()
    
    # Remove common prefixes/suffixes that might cause issues
    # This can be extended based on specific data source requirements
    
    return symbol


def validate_data_quality(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    """Validate data quality and completeness.
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        Dictionary with validation results
        
    Raises:
        DataRetrievalError: If critical data quality issues are found
    """
    validation_result = {
        'is_valid': True,
        'missing_fields': [],
        'invalid_values': {},
        'warnings': []
    }
    
    # Check for missing required fields
    for field in required_fields:
        if field not in data or data[field] is None:
            validation_result['missing_fields'].append(field)
    
    # Check for invalid values (can be extended based on specific requirements)
    for key, value in data.items():
        if isinstance(value, (int, float)) and (value < 0 and key in ['price', 'market_cap', 'volume']):
            validation_result['invalid_values'][key] = value
    
    # Determine overall validity
    if validation_result['missing_fields'] or validation_result['invalid_values']:
        validation_result['is_valid'] = False
    
    # Log warnings for data quality issues
    if validation_result['missing_fields']:
        logger.warning(f"Missing required fields: {validation_result['missing_fields']}")
    
    if validation_result['invalid_values']:
        logger.warning(f"Invalid values detected: {validation_result['invalid_values']}")
    
    return validation_result


def parse_financial_data(raw_data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
    """Parse and normalize financial data from different sources.
    
    Args:
        raw_data: Raw data from API
        data_type: Type of financial data ('security_info', 'financial_statements', etc.)
        
    Returns:
        Normalized financial data
    """
    # This function can be extended to handle specific data transformations
    # based on the data type and source format
    
    if data_type == 'security_info':
        return _parse_security_info(raw_data)
    elif data_type == 'financial_statements':
        return _parse_financial_statements(raw_data)
    elif data_type == 'technical_indicators':
        return _parse_technical_indicators(raw_data)
    else:
        return raw_data


def _parse_security_info(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse security information data."""
    # Normalize common field names across different data sources
    field_mapping = {
        'ticker': 'symbol',
        'companyName': 'name',
        'longName': 'name',
        'regularMarketPrice': 'current_price',
        'currentPrice': 'current_price',
        'marketCap': 'market_cap',
        'beta': 'beta'
    }
    
    normalized_data = {}
    for raw_key, value in raw_data.items():
        normalized_key = field_mapping.get(raw_key, raw_key)
        normalized_data[normalized_key] = value
    
    return normalized_data


def _parse_financial_statements(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse financial statements data."""
    # This can be extended to handle specific financial statement parsing
    return raw_data


def _parse_technical_indicators(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse technical indicators data."""
    # This can be extended to handle specific technical indicator parsing
    return raw_data