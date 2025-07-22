"""Financial analysis engine for calculating financial ratios and health scores.

This module provides functionality for calculating various financial ratios,
assessing company health, and calculating growth metrics from financial data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

from stock_analysis.models.data_models import (
    StockInfo, FinancialRatios, LiquidityRatios, ProfitabilityRatios,
    LeverageRatios, EfficiencyRatios, HealthScore
)
from stock_analysis.utils.exceptions import CalculationError
from stock_analysis.utils.logging import get_logger
from stock_analysis.analyzers.growth_metrics import calculate_growth_metrics

logger = get_logger(__name__)


class FinancialAnalysisEngine:
    """Engine for financial ratio calculations and health assessments."""
    
    def __init__(self):
        """Initialize the financial analysis engine."""
        logger.info("Initializing Financial Analysis Engine")
        
    def _safe_get_value(
        self,
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
    
    def calculate_financial_ratios(
        self,
        symbol: str,
        stock_info: StockInfo,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame
    ) -> FinancialRatios:
        """Calculate financial ratios from financial statements.
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Basic stock information
            income_statement: Income statement data
            balance_sheet: Balance sheet data
            cash_flow: Cash flow statement data
            
        Returns:
            FinancialRatios object containing all calculated ratios
            
        Raises:
            CalculationError: If calculations fail due to missing or invalid data
        """
        logger.info(f"Calculating financial ratios for {symbol}")
        
        try:
            # Check if we have data to work with
            if income_statement.empty or balance_sheet.empty:
                raise CalculationError(f"Missing financial statement data for {symbol}")
            
            # Get the most recent period (column) from each statement
            latest_income = income_statement.columns[0] if not income_statement.empty else None
            latest_balance = balance_sheet.columns[0] if not balance_sheet.empty else None
            latest_cash_flow = cash_flow.columns[0] if not cash_flow.empty else None
            
            # Calculate each category of ratios
            liquidity_ratios = self._calculate_liquidity_ratios(symbol, balance_sheet, latest_balance)
            profitability_ratios = self._calculate_profitability_ratios(
                symbol, income_statement, balance_sheet, latest_income, latest_balance
            )
            leverage_ratios = self._calculate_leverage_ratios(
                symbol, income_statement, balance_sheet, latest_income, latest_balance
            )
            efficiency_ratios = self._calculate_efficiency_ratios(
                symbol, income_statement, balance_sheet, latest_income, latest_balance
            )
            
            # Create and return the combined financial ratios object
            financial_ratios = FinancialRatios(
                liquidity_ratios=liquidity_ratios,
                profitability_ratios=profitability_ratios,
                leverage_ratios=leverage_ratios,
                efficiency_ratios=efficiency_ratios
            )
            
            # Validate the ratios
            financial_ratios.validate()
            return financial_ratios
            
        except Exception as e:
            if not isinstance(e, CalculationError):
                logger.error(f"Error calculating financial ratios for {symbol}: {str(e)}")
                raise CalculationError(f"Failed to calculate financial ratios for {symbol}: {str(e)}")
            else:
                raise
    
    def _calculate_liquidity_ratios(
        self,
        symbol: str,
        balance_sheet: pd.DataFrame,
        latest_period: Optional[str]
    ) -> LiquidityRatios:
        """Calculate liquidity ratios from balance sheet data.
        
        Args:
            symbol: Stock ticker symbol
            balance_sheet: Balance sheet data
            latest_period: Most recent period in the balance sheet
            
        Returns:
            LiquidityRatios object with calculated ratios
        """
        logger.debug(f"Calculating liquidity ratios for {symbol}")
        
        # Initialize with None values in case calculations fail
        current_ratio = None
        quick_ratio = None
        cash_ratio = None
        
        if latest_period and not balance_sheet.empty:
            try:
                # Extract required values from balance sheet
                current_assets = self._safe_get_value(balance_sheet, 'Total Current Assets', latest_period)
                current_liabilities = self._safe_get_value(balance_sheet, 'Total Current Liabilities', latest_period)
                cash = self._safe_get_value(balance_sheet, 'Cash And Cash Equivalents', latest_period)
                short_term_investments = self._safe_get_value(balance_sheet, 'Short Term Investments', latest_period)
                inventory = self._safe_get_value(balance_sheet, 'Inventory', latest_period)
                
                # Calculate ratios if we have the necessary data
                if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
                    current_ratio = current_assets / current_liabilities
                
                if (current_assets is not None and inventory is not None and 
                    current_liabilities is not None and current_liabilities != 0):
                    quick_ratio = (current_assets - inventory) / current_liabilities
                
                if (cash is not None and short_term_investments is not None and 
                    current_liabilities is not None and current_liabilities != 0):
                    cash_ratio = (cash + short_term_investments) / current_liabilities
                
            except Exception as e:
                logger.warning(f"Error calculating liquidity ratios for {symbol}: {str(e)}")
        
        return LiquidityRatios(
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            cash_ratio=cash_ratio
        )
    
    def _calculate_profitability_ratios(
        self,
        symbol: str,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        latest_income_period: Optional[str],
        latest_balance_period: Optional[str]
    ) -> ProfitabilityRatios:
        """Calculate profitability ratios from financial statements.
        
        Args:
            symbol: Stock ticker symbol
            income_statement: Income statement data
            balance_sheet: Balance sheet data
            latest_income_period: Most recent period in the income statement
            latest_balance_period: Most recent period in the balance sheet
            
        Returns:
            ProfitabilityRatios object with calculated ratios
        """
        logger.debug(f"Calculating profitability ratios for {symbol}")
        
        # Initialize with None values in case calculations fail
        gross_margin = None
        operating_margin = None
        net_profit_margin = None
        return_on_assets = None
        return_on_equity = None
        
        if latest_income_period and not income_statement.empty:
            try:
                # Extract required values from income statement
                revenue = self._safe_get_value(income_statement, 'Total Revenue', latest_income_period)
                gross_profit = self._safe_get_value(income_statement, 'Gross Profit', latest_income_period)
                operating_income = self._safe_get_value(income_statement, 'Operating Income', latest_income_period)
                net_income = self._safe_get_value(income_statement, 'Net Income', latest_income_period)
                
                # Calculate margin ratios if we have the necessary data
                if gross_profit is not None and revenue is not None and revenue != 0:
                    gross_margin = gross_profit / revenue
                
                if operating_income is not None and revenue is not None and revenue != 0:
                    operating_margin = operating_income / revenue
                
                if net_income is not None and revenue is not None and revenue != 0:
                    net_profit_margin = net_income / revenue
                
                # Calculate return ratios if we have balance sheet data
                if (latest_balance_period and not balance_sheet.empty and 
                    net_income is not None):
                    
                    total_assets = self._safe_get_value(balance_sheet, 'Total Assets', latest_balance_period)
                    total_equity = self._safe_get_value(balance_sheet, 'Total Stockholder Equity', latest_balance_period)
                    
                    if total_assets is not None and total_assets != 0:
                        return_on_assets = net_income / total_assets
                    
                    if total_equity is not None and total_equity != 0:
                        return_on_equity = net_income / total_equity
                
            except Exception as e:
                logger.warning(f"Error calculating profitability ratios for {symbol}: {str(e)}")
        
        return ProfitabilityRatios(
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_profit_margin=net_profit_margin,
            return_on_assets=return_on_assets,
            return_on_equity=return_on_equity
        )
    
    def _calculate_leverage_ratios(
        self,
        symbol: str,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        latest_income_period: Optional[str],
        latest_balance_period: Optional[str]
    ) -> LeverageRatios:
        """Calculate leverage ratios from financial statements.
        
        Args:
            symbol: Stock ticker symbol
            income_statement: Income statement data
            balance_sheet: Balance sheet data
            latest_income_period: Most recent period in the income statement
            latest_balance_period: Most recent period in the balance sheet
            
        Returns:
            LeverageRatios object with calculated ratios
        """
        logger.debug(f"Calculating leverage ratios for {symbol}")
        
        # Initialize with None values in case calculations fail
        debt_to_equity = None
        debt_to_assets = None
        interest_coverage = None
        
        if latest_balance_period and not balance_sheet.empty:
            try:
                # Extract required values from balance sheet
                total_debt = self._safe_get_value(balance_sheet, 'Total Debt', latest_balance_period)
                if total_debt is None:
                    # Try to calculate from long-term and short-term debt
                    long_term_debt = self._safe_get_value(balance_sheet, 'Long Term Debt', latest_balance_period)
                    short_term_debt = self._safe_get_value(balance_sheet, 'Short Term Debt', latest_balance_period)
                    if long_term_debt is not None and short_term_debt is not None:
                        total_debt = long_term_debt + short_term_debt
                
                total_equity = self._safe_get_value(balance_sheet, 'Total Stockholder Equity', latest_balance_period)
                total_assets = self._safe_get_value(balance_sheet, 'Total Assets', latest_balance_period)
                
                # Calculate debt ratios if we have the necessary data
                if total_debt is not None and total_equity is not None and total_equity != 0:
                    debt_to_equity = total_debt / total_equity
                
                if total_debt is not None and total_assets is not None and total_assets != 0:
                    debt_to_assets = total_debt / total_assets
                
                # Calculate interest coverage if we have income statement data
                if latest_income_period and not income_statement.empty:
                    operating_income = self._safe_get_value(income_statement, 'Operating Income', latest_income_period)
                    interest_expense = self._safe_get_value(income_statement, 'Interest Expense', latest_income_period)
                    
                    if (operating_income is not None and interest_expense is not None and 
                        interest_expense != 0 and abs(interest_expense) > 1e-10):
                        interest_coverage = operating_income / interest_expense
                
            except Exception as e:
                logger.warning(f"Error calculating leverage ratios for {symbol}: {str(e)}")
        
        return LeverageRatios(
            debt_to_equity=debt_to_equity,
            debt_to_assets=debt_to_assets,
            interest_coverage=interest_coverage
        )
    
    def _calculate_efficiency_ratios(
        self,
        symbol: str,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        latest_income_period: Optional[str],
        latest_balance_period: Optional[str]
    ) -> EfficiencyRatios:
        """Calculate efficiency ratios from financial statements.
        
        Args:
            symbol: Stock ticker symbol
            income_statement: Income statement data
            balance_sheet: Balance sheet data
            latest_income_period: Most recent period in the income statement
            latest_balance_period: Most recent period in the balance sheet
            
        Returns:
            EfficiencyRatios object with calculated ratios
        """
        logger.debug(f"Calculating efficiency ratios for {symbol}")
        
        # Initialize with None values in case calculations fail
        asset_turnover = None
        inventory_turnover = None
        receivables_turnover = None
        
        if (latest_income_period and latest_balance_period and 
            not income_statement.empty and not balance_sheet.empty):
            try:
                # Extract required values
                revenue = self._safe_get_value(income_statement, 'Total Revenue', latest_income_period)
                cost_of_revenue = self._safe_get_value(income_statement, 'Cost Of Revenue', latest_income_period)
                total_assets = self._safe_get_value(balance_sheet, 'Total Assets', latest_balance_period)
                inventory = self._safe_get_value(balance_sheet, 'Inventory', latest_balance_period)
                receivables = self._safe_get_value(balance_sheet, 'Net Receivables', latest_balance_period)
                
                # Calculate asset turnover
                if revenue is not None and total_assets is not None and total_assets != 0:
                    asset_turnover = revenue / total_assets
                
                # Calculate inventory turnover
                if cost_of_revenue is not None and inventory is not None and inventory != 0:
                    inventory_turnover = cost_of_revenue / inventory
                
                # Calculate receivables turnover
                if revenue is not None and receivables is not None and receivables != 0:
                    receivables_turnover = revenue / receivables
                
            except Exception as e:
                logger.warning(f"Error calculating efficiency ratios for {symbol}: {str(e)}")
        
        return EfficiencyRatios(
            asset_turnover=asset_turnover,
            inventory_turnover=inventory_turnover,
            receivables_turnover=receivables_turnover
        )
    
    def assess_company_health(
        self,
        symbol: str,
        financial_ratios: FinancialRatios
    ) -> HealthScore:
        """Assess company health based on financial ratios.
        
        Args:
            symbol: Stock ticker symbol
            financial_ratios: Financial ratios object
            
        Returns:
            HealthScore object with health assessment
            
        Raises:
            CalculationError: If health assessment fails
        """
        logger.info(f"Assessing company health for {symbol}")
        
        try:
            # Calculate individual health scores
            liquidity_health = self._calculate_liquidity_health(financial_ratios.liquidity_ratios)
            profitability_health = self._calculate_profitability_health(financial_ratios.profitability_ratios)
            financial_strength = self._calculate_financial_strength(
                financial_ratios.leverage_ratios, financial_ratios.efficiency_ratios
            )
            
            # Calculate overall score (weighted average)
            weights = {
                'liquidity': 0.25,
                'profitability': 0.35,
                'financial_strength': 0.40
            }
            
            overall_score = (
                weights['liquidity'] * liquidity_health +
                weights['profitability'] * profitability_health +
                weights['financial_strength'] * financial_strength
            )
            
            # Determine risk assessment based on overall score
            risk_assessment = self._determine_risk_level(overall_score)
            
            # Create and return health score object
            health_score = HealthScore(
                overall_score=overall_score,
                financial_strength=financial_strength,
                profitability_health=profitability_health,
                liquidity_health=liquidity_health,
                risk_assessment=risk_assessment
            )
            
            # Validate the health score
            health_score.validate()
            return health_score
            
        except Exception as e:
            logger.error(f"Error assessing company health for {symbol}: {str(e)}")
            raise CalculationError(f"Failed to assess company health for {symbol}: {str(e)}")
    
    def _calculate_liquidity_health(self, liquidity_ratios: LiquidityRatios) -> float:
        """Calculate liquidity health score.
        
        Args:
            liquidity_ratios: Liquidity ratios object
            
        Returns:
            Liquidity health score (0-100)
        """
        # Define benchmark values for each ratio
        current_ratio_benchmarks = {
            'excellent': 2.0,  # Score: 90-100
            'good': 1.5,       # Score: 70-90
            'fair': 1.0,       # Score: 50-70
            'poor': 0.8        # Score: 30-50
            # Below 0.8 is considered weak (0-30)
        }
        
        quick_ratio_benchmarks = {
            'excellent': 1.5,  # Score: 90-100
            'good': 1.0,       # Score: 70-90
            'fair': 0.8,       # Score: 50-70
            'poor': 0.5        # Score: 30-50
            # Below 0.5 is considered weak (0-30)
        }
        
        cash_ratio_benchmarks = {
            'excellent': 0.75,  # Score: 90-100
            'good': 0.5,        # Score: 70-90
            'fair': 0.3,        # Score: 50-70
            'poor': 0.2         # Score: 30-50
            # Below 0.2 is considered weak (0-30)
        }
        
        # Calculate scores for each ratio
        current_ratio_score = self._score_ratio(
            liquidity_ratios.current_ratio, current_ratio_benchmarks, higher_is_better=True
        )
        
        quick_ratio_score = self._score_ratio(
            liquidity_ratios.quick_ratio, quick_ratio_benchmarks, higher_is_better=True
        )
        
        cash_ratio_score = self._score_ratio(
            liquidity_ratios.cash_ratio, cash_ratio_benchmarks, higher_is_better=True
        )
        
        # Calculate weighted average score
        weights = {
            'current_ratio': 0.4,
            'quick_ratio': 0.4,
            'cash_ratio': 0.2
        }
        
        scores = []
        total_weight = 0
        
        if current_ratio_score is not None:
            scores.append(current_ratio_score * weights['current_ratio'])
            total_weight += weights['current_ratio']
        
        if quick_ratio_score is not None:
            scores.append(quick_ratio_score * weights['quick_ratio'])
            total_weight += weights['quick_ratio']
        
        if cash_ratio_score is not None:
            scores.append(cash_ratio_score * weights['cash_ratio'])
            total_weight += weights['cash_ratio']
        
        # Return default score if no ratios are available
        if not scores or total_weight == 0:
            return 50.0  # Neutral score
        
        # Calculate weighted average
        return sum(scores) / total_weight
    
    def _calculate_profitability_health(self, profitability_ratios: ProfitabilityRatios) -> float:
        """Calculate profitability health score.
        
        Args:
            profitability_ratios: Profitability ratios object
            
        Returns:
            Profitability health score (0-100)
        """
        # Define benchmark values for each ratio
        gross_margin_benchmarks = {
            'excellent': 0.40,  # Score: 90-100
            'good': 0.30,       # Score: 70-90
            'fair': 0.20,       # Score: 50-70
            'poor': 0.10        # Score: 30-50
            # Below 0.10 is considered weak (0-30)
        }
        
        operating_margin_benchmarks = {
            'excellent': 0.25,  # Score: 90-100
            'good': 0.15,       # Score: 70-90
            'fair': 0.10,       # Score: 50-70
            'poor': 0.05        # Score: 30-50
            # Below 0.05 is considered weak (0-30)
        }
        
        net_margin_benchmarks = {
            'excellent': 0.20,  # Score: 90-100
            'good': 0.10,       # Score: 70-90
            'fair': 0.05,       # Score: 50-70
            'poor': 0.02        # Score: 30-50
            # Below 0.02 is considered weak (0-30)
        }
        
        roe_benchmarks = {
            'excellent': 0.20,  # Score: 90-100
            'good': 0.15,       # Score: 70-90
            'fair': 0.10,       # Score: 50-70
            'poor': 0.05        # Score: 30-50
            # Below 0.05 is considered weak (0-30)
        }
        
        roa_benchmarks = {
            'excellent': 0.10,  # Score: 90-100
            'good': 0.07,       # Score: 70-90
            'fair': 0.04,       # Score: 50-70
            'poor': 0.02        # Score: 30-50
            # Below 0.02 is considered weak (0-30)
        }
        
        # Calculate scores for each ratio
        gross_margin_score = self._score_ratio(
            profitability_ratios.gross_margin, gross_margin_benchmarks, higher_is_better=True
        )
        
        operating_margin_score = self._score_ratio(
            profitability_ratios.operating_margin, operating_margin_benchmarks, higher_is_better=True
        )
        
        net_margin_score = self._score_ratio(
            profitability_ratios.net_profit_margin, net_margin_benchmarks, higher_is_better=True
        )
        
        roe_score = self._score_ratio(
            profitability_ratios.return_on_equity, roe_benchmarks, higher_is_better=True
        )
        
        roa_score = self._score_ratio(
            profitability_ratios.return_on_assets, roa_benchmarks, higher_is_better=True
        )
        
        # Calculate weighted average score
        weights = {
            'gross_margin': 0.15,
            'operating_margin': 0.20,
            'net_margin': 0.25,
            'roe': 0.25,
            'roa': 0.15
        }
        
        scores = []
        total_weight = 0
        
        if gross_margin_score is not None:
            scores.append(gross_margin_score * weights['gross_margin'])
            total_weight += weights['gross_margin']
        
        if operating_margin_score is not None:
            scores.append(operating_margin_score * weights['operating_margin'])
            total_weight += weights['operating_margin']
        
        if net_margin_score is not None:
            scores.append(net_margin_score * weights['net_margin'])
            total_weight += weights['net_margin']
        
        if roe_score is not None:
            scores.append(roe_score * weights['roe'])
            total_weight += weights['roe']
        
        if roa_score is not None:
            scores.append(roa_score * weights['roa'])
            total_weight += weights['roa']
        
        # Return default score if no ratios are available
        if not scores or total_weight == 0:
            return 50.0  # Neutral score
        
        # Calculate weighted average
        return sum(scores) / total_weight
    
    def _calculate_financial_strength(
        self,
        leverage_ratios: LeverageRatios,
        efficiency_ratios: EfficiencyRatios
    ) -> float:
        """Calculate financial strength score.
        
        Args:
            leverage_ratios: Leverage ratios object
            efficiency_ratios: Efficiency ratios object
            
        Returns:
            Financial strength score (0-100)
        """
        # Define benchmark values for leverage ratios
        debt_to_equity_benchmarks = {
            'excellent': 0.5,   # Score: 90-100
            'good': 1.0,        # Score: 70-90
            'fair': 1.5,        # Score: 50-70
            'poor': 2.0         # Score: 30-50
            # Above 2.0 is considered weak (0-30)
        }
        
        debt_to_assets_benchmarks = {
            'excellent': 0.3,   # Score: 90-100
            'good': 0.4,        # Score: 70-90
            'fair': 0.5,        # Score: 50-70
            'poor': 0.6         # Score: 30-50
            # Above 0.6 is considered weak (0-30)
        }
        
        interest_coverage_benchmarks = {
            'excellent': 8.0,   # Score: 90-100
            'good': 5.0,        # Score: 70-90
            'fair': 3.0,        # Score: 50-70
            'poor': 1.5         # Score: 30-50
            # Below 1.5 is considered weak (0-30)
        }
        
        # Define benchmark values for efficiency ratios
        asset_turnover_benchmarks = {
            'excellent': 1.0,   # Score: 90-100
            'good': 0.7,        # Score: 70-90
            'fair': 0.5,        # Score: 50-70
            'poor': 0.3         # Score: 30-50
            # Below 0.3 is considered weak (0-30)
        }
        
        # Calculate scores for each ratio
        debt_to_equity_score = self._score_ratio(
            leverage_ratios.debt_to_equity, debt_to_equity_benchmarks, higher_is_better=False
        )
        
        debt_to_assets_score = self._score_ratio(
            leverage_ratios.debt_to_assets, debt_to_assets_benchmarks, higher_is_better=False
        )
        
        interest_coverage_score = self._score_ratio(
            leverage_ratios.interest_coverage, interest_coverage_benchmarks, higher_is_better=True
        )
        
        asset_turnover_score = self._score_ratio(
            efficiency_ratios.asset_turnover, asset_turnover_benchmarks, higher_is_better=True
        )
        
        # Calculate weighted average score
        weights = {
            'debt_to_equity': 0.30,
            'debt_to_assets': 0.25,
            'interest_coverage': 0.25,
            'asset_turnover': 0.20
        }
        
        scores = []
        total_weight = 0
        
        if debt_to_equity_score is not None:
            scores.append(debt_to_equity_score * weights['debt_to_equity'])
            total_weight += weights['debt_to_equity']
        
        if debt_to_assets_score is not None:
            scores.append(debt_to_assets_score * weights['debt_to_assets'])
            total_weight += weights['debt_to_assets']
        
        if interest_coverage_score is not None:
            scores.append(interest_coverage_score * weights['interest_coverage'])
            total_weight += weights['interest_coverage']
        
        if asset_turnover_score is not None:
            scores.append(asset_turnover_score * weights['asset_turnover'])
            total_weight += weights['asset_turnover']
        
        # Return default score if no ratios are available
        if not scores or total_weight == 0:
            return 50.0  # Neutral score
        
        # Calculate weighted average
        return sum(scores) / total_weight
    
    def calculate_growth_metrics(
        self,
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
            # Use the standalone growth metrics function
            return calculate_growth_metrics(
                symbol, historical_data, income_statement, balance_sheet
            )
        except Exception as e:
            logger.error(f"Error calculating growth metrics for {symbol}: {str(e)}")
            raise CalculationError(f"Failed to calculate growth metrics for {symbol}: {str(e)}")
    
    def _score_ratio(
        self,
        ratio_value: Optional[float],
        benchmarks: Dict[str, float],
        higher_is_better: bool
    ) -> Optional[float]:
        """Score a financial ratio based on benchmark values.
        
        Args:
            ratio_value: The ratio value to score
            benchmarks: Dictionary of benchmark values for different score ranges
            higher_is_better: Whether higher values are better for this ratio
            
        Returns:
            Score between 0 and 100, or None if ratio_value is None
        """
        if ratio_value is None:
            return None
        
        # Define score ranges
        score_ranges = {
            'excellent': (90, 100),
            'good': (70, 90),
            'fair': (50, 70),
            'poor': (30, 50)
            # Below poor is weak (0-30)
        }
        
        # Sort benchmarks by value
        sorted_benchmarks = sorted(
            benchmarks.items(),
            key=lambda x: x[1],
            reverse=not higher_is_better
        )
        
        # Find the appropriate benchmark range
        for i, (level, benchmark) in enumerate(sorted_benchmarks):
            next_level = None
            if i < len(sorted_benchmarks) - 1:
                next_level = sorted_benchmarks[i + 1][0]
                next_benchmark = sorted_benchmarks[i + 1][1]
            
            # Check if ratio is better than the current benchmark
            is_better = (
                (higher_is_better and ratio_value >= benchmark) or
                (not higher_is_better and ratio_value <= benchmark)
            )
            
            # Check if ratio is worse than the next benchmark
            is_worse_than_next = False
            if next_level:
                is_worse_than_next = (
                    (higher_is_better and ratio_value < next_benchmark) or
                    (not higher_is_better and ratio_value > next_benchmark)
                )
            
            if is_better and (next_level is None or is_worse_than_next):
                # Ratio falls between current and next benchmark
                min_score, max_score = score_ranges[level]
                
                if next_level:
                    # Interpolate score within the range
                    next_min_score, next_max_score = score_ranges[next_level]
                    
                    # Calculate how far the ratio is between benchmarks
                    if higher_is_better:
                        ratio = (ratio_value - next_benchmark) / (benchmark - next_benchmark)
                    else:
                        ratio = (ratio_value - benchmark) / (next_benchmark - benchmark)
                    
                    # Interpolate score
                    return min_score + ratio * (next_min_score - min_score)
                else:
                    # Ratio is better than the best benchmark
                    return max_score
        
        # Ratio is worse than the worst benchmark
        return 15.0  # Default score for weak ratios
    
    def _determine_risk_level(self, overall_score: float) -> str:
        """Determine risk assessment based on overall health score.
        
        Args:
            overall_score: Overall health score (0-100)
            
        Returns:
            Risk assessment string ("Low", "Medium", or "High")
        """
        if overall_score >= 70:
            return "Low"
        elif overall_score >= 40:
            return "Medium"
        else:
            return "High"