"""Data quality validation and reporting module.

This module provides utilities for validating data quality, detecting anomalies,
and reporting data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime, timedelta

from stock_analysis.utils.exceptions import ValidationError, DataRetrievalError
from stock_analysis.utils.logging import get_logger, log_data_quality_issue

logger = get_logger(__name__)


class DataQualityValidator:
    """Validator for financial data quality."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize data quality validator.
        
        Args:
            strict_mode: If True, validation failures raise exceptions; otherwise, they log warnings
        """
        self.strict_mode = strict_mode
        self.logger = get_logger(f"{__name__}.DataQualityValidator")
    
    def validate_security_info(self, symbol: str, data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate security information data.
        
        Args:
            symbol: Security symbol
            data: Security information data
            
        Returns:
            Tuple of (is_valid, missing_fields, invalid_values)
            
        Raises:
            ValidationError: If validation fails and strict_mode is True
        """
        missing_fields = []
        invalid_values = {}
        
        # Required fields
        required_fields = ['symbol', 'name', 'current_price']
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        # Validate symbol match
        if 'symbol' in data and data['symbol'] != symbol:
            invalid_values['symbol'] = f"Expected {symbol}, got {data['symbol']}"
        
        # Validate numeric fields
        numeric_fields = ['current_price', 'market_cap', 'pe_ratio', 'dividend_yield', 
                         'beta', '52_week_high', '52_week_low']
        
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    # Convert to float if it's not already
                    if not isinstance(data[field], (int, float)):
                        data[field] = float(data[field])
                    
                    # Check for unreasonable values
                    if field == 'current_price' and (data[field] <= 0 or data[field] > 1000000):
                        invalid_values[field] = data[field]
                    elif field == 'market_cap' and data[field] < 0:
                        invalid_values[field] = data[field]
                    elif field == 'pe_ratio' and (data[field] < 0 or data[field] > 10000):
                        invalid_values[field] = data[field]
                    elif field == 'dividend_yield' and (data[field] < 0 or data[field] > 100):
                        invalid_values[field] = data[field]
                    elif field == 'beta' and abs(data[field]) > 10:
                        invalid_values[field] = data[field]
                except (ValueError, TypeError):
                    invalid_values[field] = data[field]
        
        # Check for data consistency
        if 'current_price' in data and '52_week_high' in data and '52_week_low' in data:
            if data['current_price'] > data['52_week_high']:
                invalid_values['current_price'] = f"Current price ({data['current_price']}) > 52-week high ({data['52_week_high']})"
            if data['current_price'] < data['52_week_low']:
                invalid_values['current_price'] = f"Current price ({data['current_price']}) < 52-week low ({data['52_week_low']})"
        
        is_valid = len(missing_fields) == 0 and len(invalid_values) == 0
        
        # Log issues
        if not is_valid:
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"Security info validation issues for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                "security_info", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"Security info validation failed for {symbol}",
                    field_name=missing_fields[0] if missing_fields else list(invalid_values.keys())[0],
                    field_value=None if not invalid_values else list(invalid_values.values())[0],
                    validation_rule="security_info_validation"
                )
        
        return is_valid, missing_fields, invalid_values
    
    def validate_historical_prices(self, symbol: str, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate historical price data.
        
        Args:
            symbol: Security symbol
            df: Historical price DataFrame
            
        Returns:
            Tuple of (is_valid, missing_fields, invalid_values)
            
        Raises:
            ValidationError: If validation fails and strict_mode is True
        """
        missing_fields = []
        invalid_values = {}
        
        # Check if DataFrame is empty
        if df is None or df.empty:
            missing_fields.append("data")
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"Historical prices validation failed: empty data for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                "historical_prices", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"Historical prices validation failed for {symbol}: empty data",
                    field_name="data",
                    validation_rule="historical_prices_validation"
                )
            
            return False, missing_fields, invalid_values
        
        # Required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                missing_fields.append(col)
        
        if missing_fields:
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"Historical prices missing required columns for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                "historical_prices", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"Historical prices validation failed for {symbol}: missing columns",
                    field_name=missing_fields[0],
                    validation_rule="historical_prices_validation"
                )
            
            return False, missing_fields, invalid_values
        
        # Check for data anomalies
        anomalies = {}
        
        # Check for negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if (df[col] < 0).any():
                negative_dates = df.index[df[col] < 0].tolist()
                anomalies[f"negative_{col.lower()}"] = negative_dates
        
        # Check for High < Low
        if ((df['High'] < df['Low']) & (~df['High'].isna()) & (~df['Low'].isna())).any():
            invalid_dates = df.index[(df['High'] < df['Low']) & (~df['High'].isna()) & (~df['Low'].isna())].tolist()
            anomalies["high_lower_than_low"] = invalid_dates
        
        # Check for unreasonable price changes (e.g., >50% in a day)
        if len(df) > 1:
            daily_returns = df['Close'].pct_change().abs()
            extreme_changes = daily_returns[daily_returns > 0.5].index.tolist()
            if extreme_changes:
                anomalies["extreme_price_changes"] = extreme_changes
        
        # Check for too many missing values
        missing_pct = df[required_columns].isna().mean().mean() * 100
        if missing_pct > 10:  # More than 10% missing data
            anomalies["high_missing_data_percentage"] = missing_pct
        
        if anomalies:
            invalid_values.update(anomalies)
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"Historical prices contain anomalies for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                "historical_prices", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"Historical prices validation failed for {symbol}: data anomalies",
                    field_name=list(anomalies.keys())[0],
                    field_value=list(anomalies.values())[0],
                    validation_rule="historical_prices_validation"
                )
            
            return False, missing_fields, invalid_values
        
        return True, [], {}
    
    def validate_financial_statements(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        statement_type: str
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate financial statement data.
        
        Args:
            symbol: Security symbol
            df: Financial statement DataFrame
            statement_type: Type of statement ('income', 'balance', 'cash')
            
        Returns:
            Tuple of (is_valid, missing_fields, invalid_values)
            
        Raises:
            ValidationError: If validation fails and strict_mode is True
        """
        missing_fields = []
        invalid_values = {}
        
        # Check if DataFrame is empty
        if df is None or df.empty:
            missing_fields.append("data")
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"{statement_type.capitalize()} statement validation failed: empty data for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                f"financial_statements_{statement_type}", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"{statement_type.capitalize()} statement validation failed for {symbol}: empty data",
                    field_name="data",
                    validation_rule=f"{statement_type}_statement_validation"
                )
            
            return False, missing_fields, invalid_values
        
        # Required fields for each statement type
        required_fields = {
            'income': ['Revenue', 'NetIncome'],
            'balance': ['TotalAssets', 'TotalLiabilities'],
            'cash': ['OperatingCashFlow']
        }
        
        # Check required fields
        for field in required_fields.get(statement_type, []):
            if field not in df.columns:
                missing_fields.append(field)
        
        if missing_fields:
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"{statement_type.capitalize()} statement missing required columns for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                f"financial_statements_{statement_type}", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"{statement_type.capitalize()} statement validation failed for {symbol}: missing columns",
                    field_name=missing_fields[0],
                    validation_rule=f"{statement_type}_statement_validation"
                )
            
            return False, missing_fields, invalid_values
        
        # Check for data anomalies
        anomalies = {}
        
        # Statement-specific validations
        if statement_type == 'income':
            # Revenue should be positive
            if 'Revenue' in df.columns and (df['Revenue'] <= 0).any():
                negative_periods = df.index[df['Revenue'] <= 0].tolist()
                anomalies["non_positive_revenue"] = negative_periods
            
            # Gross profit should be less than revenue
            if 'GrossProfit' in df.columns and 'Revenue' in df.columns:
                invalid_periods = df.index[(df['GrossProfit'] > df['Revenue']) & 
                                          (~df['GrossProfit'].isna()) & 
                                          (~df['Revenue'].isna())].tolist()
                if invalid_periods:
                    anomalies["gross_profit_exceeds_revenue"] = invalid_periods
        
        elif statement_type == 'balance':
            # Assets should equal liabilities + equity
            if all(col in df.columns for col in ['TotalAssets', 'TotalLiabilities', 'TotalShareholderEquity']):
                # Allow for small rounding differences (0.5% tolerance)
                tolerance = 0.005
                diff_pct = abs((df['TotalAssets'] - (df['TotalLiabilities'] + df['TotalShareholderEquity'])) / df['TotalAssets'])
                invalid_periods = df.index[(diff_pct > tolerance) & 
                                          (~df['TotalAssets'].isna()) & 
                                          (~df['TotalLiabilities'].isna()) & 
                                          (~df['TotalShareholderEquity'].isna())].tolist()
                if invalid_periods:
                    anomalies["balance_sheet_mismatch"] = invalid_periods
        
        elif statement_type == 'cash':
            # Operating + Investing + Financing should approximately equal net change in cash
            if all(col in df.columns for col in ['OperatingCashFlow', 'InvestingCashFlow', 'FinancingCashFlow', 'NetChangeInCash']):
                # Allow for small rounding differences (1% tolerance)
                tolerance = 0.01
                calculated = df['OperatingCashFlow'] + df['InvestingCashFlow'] + df['FinancingCashFlow']
                diff_pct = abs((calculated - df['NetChangeInCash']) / df['NetChangeInCash'].abs())
                invalid_periods = df.index[(diff_pct > tolerance) & 
                                          (~df['NetChangeInCash'].isna()) & 
                                          (~calculated.isna()) & 
                                          (df['NetChangeInCash'] != 0)].tolist()
                if invalid_periods:
                    anomalies["cash_flow_mismatch"] = invalid_periods
        
        # Check for too many missing values across all columns
        missing_pct = df.isna().mean().mean() * 100
        if missing_pct > 30:  # More than 30% missing data
            anomalies["high_missing_data_percentage"] = missing_pct
        
        if anomalies:
            invalid_values.update(anomalies)
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"{statement_type.capitalize()} statement contains anomalies for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                f"financial_statements_{statement_type}", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"{statement_type.capitalize()} statement validation failed for {symbol}: data anomalies",
                    field_name=list(anomalies.keys())[0],
                    field_value=list(anomalies.values())[0],
                    validation_rule=f"{statement_type}_statement_validation"
                )
            
            return False, missing_fields, invalid_values
        
        return True, [], {}
    
    def validate_technical_indicators(self, symbol: str, data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate technical indicators data.
        
        Args:
            symbol: Security symbol
            data: Technical indicators data
            
        Returns:
            Tuple of (is_valid, missing_fields, invalid_values)
            
        Raises:
            ValidationError: If validation fails and strict_mode is True
        """
        missing_fields = []
        invalid_values = {}
        
        # Check if data is empty
        if not data:
            missing_fields.append("data")
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"Technical indicators validation failed: empty data for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                "technical_indicators", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"Technical indicators validation failed for {symbol}: empty data",
                    field_name="data",
                    validation_rule="technical_indicators_validation"
                )
            
            return False, missing_fields, invalid_values
        
        # Validate specific indicators
        if 'rsi_14' in data:
            try:
                rsi = float(data['rsi_14'])
                if rsi < 0 or rsi > 100:
                    invalid_values['rsi_14'] = rsi
            except (ValueError, TypeError):
                invalid_values['rsi_14'] = data['rsi_14']
        
        if 'macd' in data and isinstance(data['macd'], dict):
            for key in ['macd_line', 'signal_line', 'histogram']:
                if key in data['macd'] and data['macd'][key] is not None:
                    try:
                        float(data['macd'][key])
                    except (ValueError, TypeError):
                        invalid_values[f'macd.{key}'] = data['macd'][key]
        
        # Check moving averages
        for ma in ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']:
            if ma in data:
                try:
                    ma_value = float(data[ma])
                    if ma_value <= 0:
                        invalid_values[ma] = ma_value
                except (ValueError, TypeError):
                    invalid_values[ma] = data[ma]
        
        if invalid_values:
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = f"Technical indicators contain invalid values for {symbol}"
            log_data_quality_issue(
                self.logger, 
                symbol, 
                "technical_indicators", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    f"Technical indicators validation failed for {symbol}: invalid values",
                    field_name=list(invalid_values.keys())[0],
                    field_value=list(invalid_values.values())[0],
                    validation_rule="technical_indicators_validation"
                )
            
            return False, missing_fields, invalid_values
        
        return True, [], {}
    
    def validate_market_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate market data.
        
        Args:
            data: Market data
            
        Returns:
            Tuple of (is_valid, missing_fields, invalid_values)
            
        Raises:
            ValidationError: If validation fails and strict_mode is True
        """
        missing_fields = []
        invalid_values = {}
        
        # Check if data is empty
        if not data:
            missing_fields.append("data")
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = "Market data validation failed: empty data"
            log_data_quality_issue(
                self.logger, 
                "market", 
                "market_data", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    "Market data validation failed: empty data",
                    field_name="data",
                    validation_rule="market_data_validation"
                )
            
            return False, missing_fields, invalid_values
        
        # Check required sections
        required_sections = ['indices', 'commodities', 'forex']
        for section in required_sections:
            if section not in data or not data[section]:
                missing_fields.append(section)
        
        if missing_fields:
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = "Market data missing required sections"
            log_data_quality_issue(
                self.logger, 
                "market", 
                "market_data", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    "Market data validation failed: missing sections",
                    field_name=missing_fields[0],
                    validation_rule="market_data_validation"
                )
            
            return False, missing_fields, invalid_values
        
        # Validate indices
        for index_name, index_data in data.get('indices', {}).items():
            if 'value' not in index_data:
                missing_fields.append(f"indices.{index_name}.value")
            elif index_data['value'] is not None:
                try:
                    float(index_data['value'])
                except (ValueError, TypeError):
                    invalid_values[f"indices.{index_name}.value"] = index_data['value']
        
        # Validate commodities
        for commodity_name, commodity_data in data.get('commodities', {}).items():
            if 'price' not in commodity_data:
                missing_fields.append(f"commodities.{commodity_name}.price")
            elif commodity_data['price'] is not None:
                try:
                    price = float(commodity_data['price'])
                    if price <= 0:
                        invalid_values[f"commodities.{commodity_name}.price"] = price
                except (ValueError, TypeError):
                    invalid_values[f"commodities.{commodity_name}.price"] = commodity_data['price']
        
        # Validate forex
        for pair_name, pair_data in data.get('forex', {}).items():
            if 'rate' not in pair_data:
                missing_fields.append(f"forex.{pair_name}.rate")
            elif pair_data['rate'] is not None:
                try:
                    rate = float(pair_data['rate'])
                    if rate <= 0:
                        invalid_values[f"forex.{pair_name}.rate"] = rate
                except (ValueError, TypeError):
                    invalid_values[f"forex.{pair_name}.rate"] = pair_data['rate']
        
        if missing_fields or invalid_values:
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = "Market data contains invalid or missing values"
            log_data_quality_issue(
                self.logger, 
                "market", 
                "market_data", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    "Market data validation failed: invalid or missing values",
                    field_name=missing_fields[0] if missing_fields else list(invalid_values.keys())[0],
                    field_value=None if not invalid_values else list(invalid_values.values())[0],
                    validation_rule="market_data_validation"
                )
            
            return False, missing_fields, invalid_values
        
        return True, [], {}
    
    def validate_news_item(self, news_item: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate a news item.
        
        Args:
            news_item: News item data
            
        Returns:
            Tuple of (is_valid, missing_fields, invalid_values)
            
        Raises:
            ValidationError: If validation fails and strict_mode is True
        """
        missing_fields = []
        invalid_values = {}
        
        # Required fields
        required_fields = ['title', 'source', 'published_at']
        for field in required_fields:
            if field not in news_item or news_item[field] is None:
                missing_fields.append(field)
        
        # Validate published_at date
        if 'published_at' in news_item and news_item['published_at'] is not None:
            try:
                if isinstance(news_item['published_at'], str):
                    # Try to parse the date string
                    pd.to_datetime(news_item['published_at'])
                elif not isinstance(news_item['published_at'], (datetime, pd.Timestamp)):
                    invalid_values['published_at'] = news_item['published_at']
            except (ValueError, TypeError):
                invalid_values['published_at'] = news_item['published_at']
        
        # Validate sentiment if present
        if 'sentiment' in news_item and news_item['sentiment'] is not None:
            try:
                sentiment = float(news_item['sentiment'])
                if sentiment < -1.0 or sentiment > 1.0:
                    invalid_values['sentiment'] = sentiment
            except (ValueError, TypeError):
                invalid_values['sentiment'] = news_item['sentiment']
        
        # Validate impact if present
        if 'impact' in news_item and news_item['impact'] is not None:
            if news_item['impact'] not in ['high', 'medium', 'low']:
                invalid_values['impact'] = news_item['impact']
        
        if missing_fields or invalid_values:
            is_valid = False
            
            severity = "error" if self.strict_mode else "warning"
            issue_description = "News item contains invalid or missing values"
            
            # Use title as identifier if available, otherwise use source
            identifier = news_item.get('title', news_item.get('source', 'unknown'))
            
            log_data_quality_issue(
                self.logger, 
                identifier, 
                "news_item", 
                issue_description,
                severity, 
                missing_fields, 
                invalid_values
            )
            
            if self.strict_mode:
                raise ValidationError(
                    "News item validation failed: invalid or missing values",
                    field_name=missing_fields[0] if missing_fields else list(invalid_values.keys())[0],
                    field_value=None if not invalid_values else list(invalid_values.values())[0],
                    validation_rule="news_item_validation"
                )
            
            return False, missing_fields, invalid_values
        
        return True, [], {}
    
    def validate_news_items(self, news_items: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate a list of news items.
        
        Args:
            news_items: List of news items
            
        Returns:
            Tuple of (all_valid, valid_items, invalid_items)
        """
        if not news_items:
            return True, [], []
        
        valid_items = []
        invalid_items = []
        
        for item in news_items:
            is_valid, _, _ = self.validate_news_item(item)
            if is_valid:
                valid_items.append(item)
            else:
                invalid_items.append(item)
        
        all_valid = len(invalid_items) == 0
        return all_valid, valid_items, invalid_items


# Global validator instance
_global_validator: Optional[DataQualityValidator] = None


def get_data_quality_validator(strict_mode: bool = False) -> DataQualityValidator:
    """Get the global data quality validator instance.
    
    Args:
        strict_mode: If True, create a new validator in strict mode
        
    Returns:
        DataQualityValidator instance
    """
    global _global_validator
    if _global_validator is None or strict_mode != _global_validator.strict_mode:
        _global_validator = DataQualityValidator(strict_mode=strict_mode)
    return _global_validator


def validate_data_source_response(
    source_name: str,
    data_type: str,
    response: Any,
    symbol: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Validate response from a data source.
    
    Args:
        source_name: Name of the data source
        data_type: Type of data ('security_info', 'historical_prices', etc.)
        response: Response data to validate
        symbol: Security symbol (if applicable)
        
    Returns:
        Tuple of (is_valid, validation_details)
    """
    validator = get_data_quality_validator(strict_mode=False)
    logger = get_logger(__name__)
    
    validation_details = {
        'source': source_name,
        'data_type': data_type,
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'is_valid': False,
        'missing_fields': [],
        'invalid_values': {},
        'warnings': []
    }
    
    try:
        if data_type == 'security_info' and symbol:
            is_valid, missing, invalid = validator.validate_security_info(symbol, response)
        elif data_type == 'historical_prices' and symbol:
            is_valid, missing, invalid = validator.validate_historical_prices(symbol, response)
        elif data_type.startswith('financial_statements_') and symbol:
            statement_type = data_type.split('_')[-1]
            is_valid, missing, invalid = validator.validate_financial_statements(symbol, response, statement_type)
        elif data_type == 'technical_indicators' and symbol:
            is_valid, missing, invalid = validator.validate_technical_indicators(symbol, response)
        elif data_type == 'market_data':
            is_valid, missing, invalid = validator.validate_market_data(response)
        elif data_type == 'news_items':
            if isinstance(response, list):
                is_valid, valid_items, invalid_items = validator.validate_news_items(response)
                missing = []
                invalid = {'invalid_items_count': len(invalid_items)}
                if invalid_items:
                    validation_details['warnings'].append(f"{len(invalid_items)} news items failed validation")
            else:
                is_valid, missing, invalid = False, ['expected_list'], {'actual_type': type(response).__name__}
        else:
            # For unsupported data types, just check if response is not None or empty
            is_valid = response is not None
            if isinstance(response, (list, dict)) and not response:
                is_valid = False
            missing = [] if is_valid else ['data']
            invalid = {}
        
        validation_details['is_valid'] = is_valid
        validation_details['missing_fields'] = missing
        validation_details['invalid_values'] = invalid
        
        if not is_valid:
            logger.warning(
                f"Data validation failed for {source_name} {data_type}" +
                (f" ({symbol})" if symbol else "") +
                f": {len(missing)} missing fields, {len(invalid)} invalid values"
            )
        
        return is_valid, validation_details
        
    except Exception as e:
        logger.error(f"Error during data validation: {str(e)}")
        validation_details['is_valid'] = False
        validation_details['warnings'].append(f"Validation error: {str(e)}")
        return False, validation_details


def generate_data_quality_report(
    symbol: str,
    data_sources: List[str],
    data_types: List[str]
) -> Dict[str, Any]:
    """Generate a comprehensive data quality report.
    
    Args:
        symbol: Security symbol
        data_sources: List of data sources to check
        data_types: List of data types to check
        
    Returns:
        Data quality report
    """
    from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
    
    service = FinancialDataIntegrationService()
    validator = get_data_quality_validator(strict_mode=False)
    logger = get_logger(__name__)
    
    report = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'overall_quality_score': 0.0,
        'data_sources': data_sources,
        'data_types': data_types,
        'validation_results': {},
        'issues_summary': {
            'critical': 0,
            'major': 0,
            'minor': 0
        },
        'recommendations': []
    }
    
    # Track quality scores for each data type
    quality_scores = []
    
    # Check each data type
    for data_type in data_types:
        report['validation_results'][data_type] = {}
        
        try:
            # Retrieve and validate data from each source
            for source_name in data_sources:
                adapter = service.get_adapter_by_name(source_name)
                if not adapter:
                    report['validation_results'][data_type][source_name] = {
                        'status': 'error',
                        'message': f"Adapter '{source_name}' not found"
                    }
                    continue
                
                try:
                    # Retrieve data based on data type
                    if data_type == 'security_info':
                        data = adapter.get_security_info(symbol)
                    elif data_type == 'historical_prices':
                        data = adapter.get_historical_prices(symbol)
                    elif data_type.startswith('financial_statements_'):
                        statement_type = data_type.split('_')[-1]
                        data = adapter.get_financial_statements(symbol, statement_type)
                    elif data_type == 'technical_indicators':
                        data = adapter.get_technical_indicators(symbol)
                    elif data_type == 'market_data':
                        data = adapter.get_market_data()
                    elif data_type == 'news':
                        data = adapter.get_news(symbol=symbol)
                    else:
                        report['validation_results'][data_type][source_name] = {
                            'status': 'error',
                            'message': f"Unsupported data type: {data_type}"
                        }
                        continue
                    
                    # Validate the data
                    is_valid, validation_details = validate_data_source_response(
                        source_name, data_type, data, symbol
                    )
                    
                    # Calculate quality score (0.0 to 1.0)
                    if is_valid:
                        quality_score = 1.0
                    else:
                        # Deduct points for missing fields and invalid values
                        missing_penalty = len(validation_details['missing_fields']) * 0.1
                        invalid_penalty = len(validation_details['invalid_values']) * 0.1
                        quality_score = max(0.0, 1.0 - missing_penalty - invalid_penalty)
                    
                    quality_scores.append(quality_score)
                    
                    # Categorize issues
                    if quality_score < 0.6:
                        report['issues_summary']['critical'] += 1
                    elif quality_score < 0.8:
                        report['issues_summary']['major'] += 1
                    elif quality_score < 1.0:
                        report['issues_summary']['minor'] += 1
                    
                    # Add validation results to report
                    report['validation_results'][data_type][source_name] = {
                        'status': 'success',
                        'is_valid': is_valid,
                        'quality_score': quality_score,
                        'details': validation_details
                    }
                    
                except Exception as e:
                    logger.error(f"Error retrieving {data_type} from {source_name}: {str(e)}")
                    report['validation_results'][data_type][source_name] = {
                        'status': 'error',
                        'message': str(e)
                    }
                    report['issues_summary']['critical'] += 1
        
        except Exception as e:
            logger.error(f"Error processing data type {data_type}: {str(e)}")
            report['validation_results'][data_type] = {
                'status': 'error',
                'message': str(e)
            }
    
    # Calculate overall quality score
    if quality_scores:
        report['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
    
    # Generate recommendations
    if report['issues_summary']['critical'] > 0:
        report['recommendations'].append(
            "Critical data quality issues detected. Consider using alternative data sources."
        )
    
    if report['overall_quality_score'] < 0.7:
        report['recommendations'].append(
            "Overall data quality is poor. Implement more robust validation and fallback mechanisms."
        )
    
    # Add source-specific recommendations
    for data_type, sources in report['validation_results'].items():
        for source_name, result in sources.items():
            if isinstance(result, dict) and result.get('status') == 'success' and not result.get('is_valid', True):
                report['recommendations'].append(
                    f"Review {source_name} adapter for {data_type} data quality issues."
                )
    
    return report