"""Growth metrics calculation for financial analysis.

This module provides functionality for calculating growth metrics from historical data
and financial statements.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

from stock_analysis.utils.exceptions import CalculationError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_growth_metrics(
    symbol: str,
    historical_data: pd.DataFrame,
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame
) -> Dict[str, float]:
    """Calculate growth metrics from historical data and financial statements.
    
    Args:
        symbol: Stock ticker symbol
        historical_data: Historical price data
        income_statement: Income statement data
        balance_sheet: Balance sheet data
        
    Returns:
        Dictionary containing growth metrics
        
    Raises:
        CalculationError: If calculations fail due to missing or invalid data
    """
    logger.info(f"Calculating growth metrics for {symbol}")
    
    try:
        growth_metrics = {}
        
        # Calculate revenue growth
        if not income_statement.empty and len(income_statement.columns) >= 2:
            latest_period = income_statement.columns[0]
            previous_period = income_statement.columns[1]
            
            latest_revenue = _safe_get_value(income_statement, 'Total Revenue', latest_period)
            previous_revenue = _safe_get_value(income_statement, 'Total Revenue', previous_period)
            
            if latest_revenue is not None and previous_revenue is not None and previous_revenue != 0:
                growth_metrics['revenue_growth'] = (latest_revenue - previous_revenue) / previous_revenue
            else:
                growth_metrics['revenue_growth'] = None
        else:
            growth_metrics['revenue_growth'] = None
        
        # Calculate earnings growth
        if not income_statement.empty and len(income_statement.columns) >= 2:
            latest_period = income_statement.columns[0]
            previous_period = income_statement.columns[1]
            
            latest_earnings = _safe_get_value(income_statement, 'Net Income', latest_period)
            previous_earnings = _safe_get_value(income_statement, 'Net Income', previous_period)
            
            if latest_earnings is not None and previous_earnings is not None and previous_earnings != 0:
                growth_metrics['earnings_growth'] = (latest_earnings - previous_earnings) / previous_earnings
            else:
                growth_metrics['earnings_growth'] = None
        else:
            growth_metrics['earnings_growth'] = None
        
        # Calculate asset growth
        if not balance_sheet.empty and len(balance_sheet.columns) >= 2:
            latest_period = balance_sheet.columns[0]
            previous_period = balance_sheet.columns[1]
            
            latest_assets = _safe_get_value(balance_sheet, 'Total Assets', latest_period)
            previous_assets = _safe_get_value(balance_sheet, 'Total Assets', previous_period)
            
            if latest_assets is not None and previous_assets is not None and previous_assets != 0:
                growth_metrics['asset_growth'] = (latest_assets - previous_assets) / previous_assets
            else:
                growth_metrics['asset_growth'] = None
        else:
            growth_metrics['asset_growth'] = None
        
        # Calculate equity growth
        if not balance_sheet.empty and len(balance_sheet.columns) >= 2:
            latest_period = balance_sheet.columns[0]
            previous_period = balance_sheet.columns[1]
            
            latest_equity = _safe_get_value(balance_sheet, 'Total Stockholder Equity', latest_period)
            previous_equity = _safe_get_value(balance_sheet, 'Total Stockholder Equity', previous_period)
            
            if latest_equity is not None and previous_equity is not None and previous_equity != 0:
                growth_metrics['equity_growth'] = (latest_equity - previous_equity) / previous_equity
            else:
                growth_metrics['equity_growth'] = None
        else:
            growth_metrics['equity_growth'] = None
        
        # Calculate price growth (if historical data is available)
        if not historical_data.empty and len(historical_data) > 1:
            # Get the first and last closing prices
            latest_price = historical_data['Close'].iloc[0]
            earliest_price = historical_data['Close'].iloc[-1]
            
            if latest_price is not None and earliest_price is not None and earliest_price != 0:
                # Calculate annualized growth rate
                days = (historical_data.index[0] - historical_data.index[-1]).days
                if days > 0:
                    years = days / 365.0
                    if years > 0:
                        growth_metrics['price_cagr'] = ((latest_price / earliest_price) ** (1 / years)) - 1
                    else:
                        growth_metrics['price_cagr'] = None
                else:
                    growth_metrics['price_cagr'] = None
            else:
                growth_metrics['price_cagr'] = None
        else:
            growth_metrics['price_cagr'] = None
        
        # Calculate EPS growth
        if not income_statement.empty and len(income_statement.columns) >= 2:
            latest_period = income_statement.columns[0]
            previous_period = income_statement.columns[1]
            
            latest_eps = _safe_get_value(income_statement, 'EPS - Basic', latest_period)
            previous_eps = _safe_get_value(income_statement, 'EPS - Basic', previous_period)
            
            # If EPS is not directly available, try to calculate it
            if latest_eps is None or previous_eps is None:
                latest_net_income = _safe_get_value(income_statement, 'Net Income', latest_period)
                previous_net_income = _safe_get_value(income_statement, 'Net Income', previous_period)
                
                # Try to get shares outstanding from balance sheet
                latest_shares = _safe_get_value(balance_sheet, 'Common Stock Shares Outstanding', latest_period)
                previous_shares = _safe_get_value(balance_sheet, 'Common Stock Shares Outstanding', previous_period)
                
                if (latest_net_income is not None and previous_net_income is not None and
                    latest_shares is not None and previous_shares is not None and
                    latest_shares != 0 and previous_shares != 0):
                    latest_eps = latest_net_income / latest_shares
                    previous_eps = previous_net_income / previous_shares
            
            if latest_eps is not None and previous_eps is not None and previous_eps != 0:
                growth_metrics['eps_growth'] = (latest_eps - previous_eps) / previous_eps
            else:
                growth_metrics['eps_growth'] = None
        else:
            growth_metrics['eps_growth'] = None
        
        # Calculate dividend growth if available
        if not income_statement.empty and len(income_statement.columns) >= 2:
            latest_period = income_statement.columns[0]
            previous_period = income_statement.columns[1]
            
            latest_dividend = _safe_get_value(income_statement, 'Dividends Paid', latest_period)
            previous_dividend = _safe_get_value(income_statement, 'Dividends Paid', previous_period)
            
            if latest_dividend is not None and previous_dividend is not None and previous_dividend != 0:
                growth_metrics['dividend_growth'] = (latest_dividend - previous_dividend) / previous_dividend
            else:
                growth_metrics['dividend_growth'] = None
        else:
            growth_metrics['dividend_growth'] = None
        
        return growth_metrics
        
    except Exception as e:
        logger.error(f"Error calculating growth metrics for {symbol}: {str(e)}")
        raise CalculationError(f"Failed to calculate growth metrics for {symbol}: {str(e)}")


def _safe_get_value(
    df: pd.DataFrame,
    row_name: str,
    column_name: str
) -> Optional[float]:
    """Safely get a value from a DataFrame.
    
    Args:
        df: DataFrame to get value from
        row_name: Row name (index)
        column_name: Column name
        
    Returns:
        Value if found, None otherwise
    """
    try:
        if df.empty or row_name not in df.index or column_name not in df.columns:
            return None
        
        value = df.loc[row_name, column_name]
        
        # Check if value is NaN or infinite
        if pd.isna(value) or np.isinf(value):
            return None
        
        return float(value)
    except Exception as e:
        logger.debug(f"Error getting value for {row_name}, {column_name}: {str(e)}")
        return None