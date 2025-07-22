"""Valuation engine for calculating fair value using multiple models.

This module provides functionality for calculating fair value using various
valuation methods including DCF, peer comparison, and relative valuation ratios.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

from stock_analysis.models.data_models import StockInfo, FairValueResult
from stock_analysis.utils.exceptions import CalculationError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class ValuationEngine:
    """Engine for calculating fair value using multiple valuation models."""
    
    def __init__(self):
        """Initialize the valuation engine."""
        logger.info("Initializing Valuation Engine")
        
        # Default assumptions for DCF model
        self.default_terminal_growth_rate = 0.025  # 2.5%
        self.default_discount_rate = 0.10  # 10%
        self.default_projection_years = 5
        
        # Default assumptions for peer comparison
        self.peer_weight = 0.3
        self.dcf_weight = 0.7
    
    def calculate_fair_value(
        self,
        symbol: str,
        stock_info: StockInfo,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        peer_data: Optional[List[StockInfo]] = None
    ) -> FairValueResult:
        """Calculate fair value using multiple valuation models.
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Basic stock information
            income_statement: Income statement data
            balance_sheet: Balance sheet data
            cash_flow: Cash flow statement data
            peer_data: List of peer company stock information
            
        Returns:
            FairValueResult object with valuation results
            
        Raises:
            CalculationError: If valuation calculations fail
        """
        logger.info(f"Calculating fair value for {symbol}")
        
        try:
            # Calculate DCF value
            dcf_value = self.calculate_dcf_value(
                symbol, stock_info, income_statement, balance_sheet, cash_flow
            )
            
            # Calculate peer comparison value
            peer_comparison_value = None
            if peer_data:
                peer_comparison_value = self.calculate_peer_comparison_value(
                    symbol, stock_info, peer_data
                )
            
            # Calculate PEG ratio and other relative valuation metrics
            peg_ratio = self.calculate_peg_ratio(symbol, stock_info, income_statement)
            
            # Aggregate fair value and generate recommendation
            fair_value_result = self._aggregate_fair_value(
                symbol, stock_info, dcf_value, peer_comparison_value, peg_ratio
            )
            
            # Validate the result
            fair_value_result.validate()
            return fair_value_result
            
        except Exception as e:
            logger.error(f"Error calculating fair value for {symbol}: {str(e)}")
            raise CalculationError(f"Failed to calculate fair value for {symbol}: {str(e)}")
    
    def calculate_dcf_value(
        self,
        symbol: str,
        stock_info: StockInfo,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame
    ) -> Optional[float]:
        """Calculate fair value using Discounted Cash Flow (DCF) model.
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Basic stock information
            income_statement: Income statement data
            balance_sheet: Balance sheet data
            cash_flow: Cash flow statement data
            
        Returns:
            DCF fair value per share, or None if calculation fails
        """
        logger.debug(f"Calculating DCF value for {symbol}")
        
        try:
            # Get free cash flow data
            free_cash_flows = self._extract_free_cash_flows(cash_flow)
            if not free_cash_flows or len(free_cash_flows) < 2:
                logger.warning(f"Insufficient cash flow data for DCF calculation: {symbol}")
                return None
            
            # Calculate growth rate from historical data
            growth_rate = self._calculate_cash_flow_growth_rate(free_cash_flows)
            if growth_rate is None:
                logger.warning(f"Could not calculate growth rate for {symbol}")
                return None
            
            # Project future cash flows
            projected_cash_flows = self._project_cash_flows(
                free_cash_flows[-1], growth_rate, self.default_projection_years
            )
            
            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(
                projected_cash_flows[-1], self.default_terminal_growth_rate
            )
            
            # Discount cash flows to present value
            present_value = self._discount_cash_flows(
                projected_cash_flows, terminal_value, self.default_discount_rate
            )
            
            # Calculate per-share value
            shares_outstanding = self._get_shares_outstanding(balance_sheet, stock_info)
            if shares_outstanding is None or shares_outstanding <= 0:
                logger.warning(f"Could not determine shares outstanding for {symbol}")
                return None
            
            dcf_value_per_share = present_value / shares_outstanding
            
            logger.debug(f"DCF value for {symbol}: ${dcf_value_per_share:.2f}")
            return dcf_value_per_share
            
        except Exception as e:
            logger.warning(f"Error calculating DCF value for {symbol}: {str(e)}")
            return None
    
    def calculate_peer_comparison_value(
        self,
        symbol: str,
        stock_info: StockInfo,
        peer_data: List[StockInfo]
    ) -> Optional[float]:
        """Calculate fair value using peer comparison method.
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Basic stock information
            peer_data: List of peer company stock information
            
        Returns:
            Peer comparison fair value, or None if calculation fails
        """
        logger.debug(f"Calculating peer comparison value for {symbol}")
        
        try:
            if not peer_data:
                logger.warning(f"No peer data available for {symbol}")
                return None
            
            # Calculate average P/E ratio of peers
            peer_pe_ratios = [peer.pe_ratio for peer in peer_data if peer.pe_ratio is not None]
            if not peer_pe_ratios:
                logger.warning(f"No P/E ratios available for peers of {symbol}")
                return None
            
            avg_peer_pe = np.mean(peer_pe_ratios)
            median_peer_pe = np.median(peer_pe_ratios)
            
            # Use median to reduce impact of outliers
            peer_pe_multiple = median_peer_pe
            
            # Get earnings per share (approximate from market cap and P/E)
            if stock_info.pe_ratio is not None and stock_info.pe_ratio > 0:
                current_eps = stock_info.current_price / stock_info.pe_ratio
                peer_value = current_eps * peer_pe_multiple
            else:
                logger.warning(f"No P/E ratio available for {symbol}")
                return None
            
            logger.debug(f"Peer comparison value for {symbol}: ${peer_value:.2f}")
            return peer_value
            
        except Exception as e:
            logger.warning(f"Error calculating peer comparison value for {symbol}: {str(e)}")
            return None
    
    def calculate_peg_ratio(
        self,
        symbol: str,
        stock_info: StockInfo,
        income_statement: pd.DataFrame
    ) -> Optional[float]:
        """Calculate PEG (Price/Earnings to Growth) ratio.
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Basic stock information
            income_statement: Income statement data
            
        Returns:
            PEG ratio, or None if calculation fails
        """
        logger.debug(f"Calculating PEG ratio for {symbol}")
        
        try:
            if stock_info.pe_ratio is None or stock_info.pe_ratio <= 0:
                logger.warning(f"No valid P/E ratio for PEG calculation: {symbol}")
                return None
            
            # Calculate earnings growth rate from income statement
            earnings_growth_rate = self._calculate_earnings_growth_rate(income_statement)
            if earnings_growth_rate is None or earnings_growth_rate <= 0:
                logger.warning(f"No valid earnings growth rate for PEG calculation: {symbol}")
                return None
            
            # Convert growth rate to percentage
            earnings_growth_percentage = earnings_growth_rate * 100
            
            # Calculate PEG ratio
            peg_ratio = stock_info.pe_ratio / earnings_growth_percentage
            
            logger.debug(f"PEG ratio for {symbol}: {peg_ratio:.2f}")
            return peg_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating PEG ratio for {symbol}: {str(e)}")
            return None
    
    def _extract_free_cash_flows(self, cash_flow: pd.DataFrame) -> List[float]:
        """Extract free cash flow values from cash flow statement.
        
        Args:
            cash_flow: Cash flow statement data
            
        Returns:
            List of free cash flow values (most recent first)
        """
        try:
            if cash_flow.empty:
                return []
            
            # Try to find free cash flow directly
            free_cash_flow_rows = [
                'Free Cash Flow',
                'FreeCashFlow',
                'Free_Cash_Flow'
            ]
            
            for row_name in free_cash_flow_rows:
                if row_name in cash_flow.index:
                    values = []
                    for col in cash_flow.columns:
                        value = cash_flow.loc[row_name, col]
                        if pd.notna(value) and not np.isinf(value):
                            values.append(float(value))
                    return values
            
            # If not found, calculate from operating cash flow and capex
            operating_cf_rows = [
                'Operating Cash Flow',
                'Total Cash From Operating Activities',
                'Cash From Operating Activities'
            ]
            
            capex_rows = [
                'Capital Expenditures',
                'Capital Expenditure',
                'Capex',
                'Property Plant Equipment'
            ]
            
            operating_cf = None
            capex = None
            
            # Find operating cash flow
            for row_name in operating_cf_rows:
                if row_name in cash_flow.index:
                    operating_cf = cash_flow.loc[row_name]
                    break
            
            # Find capital expenditures
            for row_name in capex_rows:
                if row_name in cash_flow.index:
                    capex = cash_flow.loc[row_name]
                    break
            
            if operating_cf is not None and capex is not None:
                free_cash_flows = []
                for col in cash_flow.columns:
                    ocf_val = operating_cf[col] if pd.notna(operating_cf[col]) else 0
                    capex_val = capex[col] if pd.notna(capex[col]) else 0
                    # Capex is usually negative, so we add it (subtract the absolute value)
                    fcf = ocf_val + capex_val
                    if not np.isinf(fcf):
                        free_cash_flows.append(float(fcf))
                return free_cash_flows
            
            return []
            
        except Exception as e:
            logger.warning(f"Error extracting free cash flows: {str(e)}")
            return [] 
   
    def _calculate_cash_flow_growth_rate(self, free_cash_flows: List[float]) -> Optional[float]:
        """Calculate growth rate from historical free cash flows.
        
        Args:
            free_cash_flows: List of free cash flow values (most recent first)
            
        Returns:
            Average growth rate, or None if calculation fails
        """
        try:
            if len(free_cash_flows) < 2:
                return None
            
            # Calculate year-over-year growth rates
            growth_rates = []
            for i in range(len(free_cash_flows) - 1):
                current = free_cash_flows[i]
                previous = free_cash_flows[i + 1]
                
                if previous != 0 and current > 0 and previous > 0:
                    growth_rate = (current - previous) / abs(previous)
                    # Cap extreme growth rates
                    growth_rate = max(-0.5, min(growth_rate, 1.0))
                    growth_rates.append(growth_rate)
            
            if not growth_rates:
                return None
            
            # Return median growth rate to reduce impact of outliers
            return float(np.median(growth_rates))
            
        except Exception as e:
            logger.warning(f"Error calculating cash flow growth rate: {str(e)}")
            return None
    
    def _project_cash_flows(
        self, 
        base_cash_flow: float, 
        growth_rate: float, 
        years: int
    ) -> List[float]:
        """Project future cash flows based on growth rate.
        
        Args:
            base_cash_flow: Starting cash flow value
            growth_rate: Annual growth rate
            years: Number of years to project
            
        Returns:
            List of projected cash flows
        """
        projected_flows = []
        current_flow = base_cash_flow
        
        # Cap growth rate to reasonable bounds
        capped_growth_rate = max(-0.2, min(growth_rate, 0.3))
        
        for year in range(1, years + 1):
            # Apply declining growth rate over time
            adjusted_growth = capped_growth_rate * (0.9 ** (year - 1))
            current_flow = current_flow * (1 + adjusted_growth)
            projected_flows.append(current_flow)
        
        return projected_flows
    
    def _calculate_terminal_value(
        self, 
        final_cash_flow: float, 
        terminal_growth_rate: float
    ) -> float:
        """Calculate terminal value using perpetual growth model.
        
        Args:
            final_cash_flow: Cash flow in the final projection year
            terminal_growth_rate: Long-term growth rate
            
        Returns:
            Terminal value
        """
        # Terminal value = FCF * (1 + g) / (r - g)
        # Where g is terminal growth rate and r is discount rate
        terminal_cash_flow = final_cash_flow * (1 + terminal_growth_rate)
        terminal_value = terminal_cash_flow / (self.default_discount_rate - terminal_growth_rate)
        return terminal_value
    
    def _discount_cash_flows(
        self, 
        projected_cash_flows: List[float], 
        terminal_value: float, 
        discount_rate: float
    ) -> float:
        """Discount projected cash flows and terminal value to present value.
        
        Args:
            projected_cash_flows: List of projected cash flows
            terminal_value: Terminal value
            discount_rate: Discount rate (WACC)
            
        Returns:
            Total present value
        """
        present_value = 0.0
        
        # Discount projected cash flows
        for year, cash_flow in enumerate(projected_cash_flows, 1):
            pv = cash_flow / ((1 + discount_rate) ** year)
            present_value += pv
        
        # Discount terminal value
        terminal_pv = terminal_value / ((1 + discount_rate) ** len(projected_cash_flows))
        present_value += terminal_pv
        
        return present_value
    
    def _get_shares_outstanding(
        self, 
        balance_sheet: pd.DataFrame, 
        stock_info: StockInfo
    ) -> Optional[float]:
        """Get shares outstanding from balance sheet or calculate from market cap.
        
        Args:
            balance_sheet: Balance sheet data
            stock_info: Basic stock information
            
        Returns:
            Number of shares outstanding, or None if not available
        """
        try:
            # Try to find shares outstanding in balance sheet
            shares_rows = [
                'Ordinary Shares Number',
                'Common Stock Shares Outstanding',
                'Shares Outstanding',
                'Common Shares Outstanding'
            ]
            
            if not balance_sheet.empty:
                latest_period = balance_sheet.columns[0]
                for row_name in shares_rows:
                    if row_name in balance_sheet.index:
                        shares = balance_sheet.loc[row_name, latest_period]
                        if pd.notna(shares) and shares > 0:
                            return float(shares)
            
            # Calculate from market cap and current price
            if stock_info.market_cap > 0 and stock_info.current_price > 0:
                shares_outstanding = stock_info.market_cap / stock_info.current_price
                return shares_outstanding
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting shares outstanding: {str(e)}")
            return None
    
    def _calculate_earnings_growth_rate(self, income_statement: pd.DataFrame) -> Optional[float]:
        """Calculate earnings growth rate from income statement.
        
        Args:
            income_statement: Income statement data
            
        Returns:
            Average earnings growth rate, or None if calculation fails
        """
        try:
            if income_statement.empty or len(income_statement.columns) < 2:
                return None
            
            # Find net income row
            net_income_rows = [
                'Net Income',
                'Net Income Common Stockholders',
                'Net Income Applicable To Common Shares'
            ]
            
            net_income_data = None
            for row_name in net_income_rows:
                if row_name in income_statement.index:
                    net_income_data = income_statement.loc[row_name]
                    break
            
            if net_income_data is None:
                return None
            
            # Calculate year-over-year growth rates
            growth_rates = []
            for i in range(len(net_income_data) - 1):
                current = net_income_data.iloc[i]
                previous = net_income_data.iloc[i + 1]
                
                if (pd.notna(current) and pd.notna(previous) and 
                    previous != 0 and current > 0 and previous > 0):
                    growth_rate = (current - previous) / abs(previous)
                    # Cap extreme growth rates
                    growth_rate = max(-0.5, min(growth_rate, 2.0))
                    growth_rates.append(growth_rate)
            
            if not growth_rates:
                return None
            
            # Return median growth rate
            return float(np.median(growth_rates))
            
        except Exception as e:
            logger.warning(f"Error calculating earnings growth rate: {str(e)}")
            return None
    
    def _aggregate_fair_value(
        self,
        symbol: str,
        stock_info: StockInfo,
        dcf_value: Optional[float],
        peer_comparison_value: Optional[float],
        peg_ratio: Optional[float]
    ) -> FairValueResult:
        """Aggregate fair value from different models and generate recommendation.
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Basic stock information
            dcf_value: DCF fair value
            peer_comparison_value: Peer comparison fair value
            peg_ratio: PEG ratio
            
        Returns:
            FairValueResult with aggregated valuation and recommendation
        """
        logger.debug(f"Aggregating fair value for {symbol}")
        
        # Log individual valuation results for debugging
        logger.info(f"Valuation results for {symbol}: DCF=${dcf_value}, Peer=${peer_comparison_value}, PEG={peg_ratio}")
        
        # Calculate weighted average fair value
        values = []
        weights = []
        
        # Include DCF value if it exists, even if negative (indicates financial distress)
        if dcf_value is not None:
            values.append(dcf_value)
            weights.append(self.dcf_weight)
            if dcf_value < 0:
                logger.warning(f"Negative DCF value for {symbol}: ${dcf_value:.2f} - indicates potential financial distress")
        
        if peer_comparison_value is not None and peer_comparison_value > 0:
            values.append(peer_comparison_value)
            weights.append(self.peer_weight)
        
        # Calculate average fair value
        if values:
            total_weight = sum(weights)
            average_fair_value = sum(v * w for v, w in zip(values, weights)) / total_weight
            logger.info(f"Weighted average fair value for {symbol}: ${average_fair_value:.2f} (using {len(values)} models)")
        else:
            # Fallback to current price if no models worked
            average_fair_value = stock_info.current_price
            logger.warning(f"No valuation models succeeded for {symbol}, using current price as fair value")
        
        # Generate recommendation based on fair value vs current price
        recommendation, confidence_level = self._generate_recommendation(
            stock_info.current_price, average_fair_value, peg_ratio, len(values)
        )
        
        return FairValueResult(
            current_price=stock_info.current_price,
            dcf_value=dcf_value,
            peer_comparison_value=peer_comparison_value,
            average_fair_value=average_fair_value,
            recommendation=recommendation,
            confidence_level=confidence_level
        )
    
    def _generate_recommendation(
        self,
        current_price: float,
        fair_value: float,
        peg_ratio: Optional[float],
        num_models: int
    ) -> Tuple[str, float]:
        """Generate buy/hold/sell recommendation based on valuation metrics.
        
        Args:
            current_price: Current stock price
            fair_value: Calculated fair value
            peg_ratio: PEG ratio
            num_models: Number of valuation models used
            
        Returns:
            Tuple of (recommendation, confidence_level)
        """
        # Handle negative fair value (indicates financial distress)
        if fair_value <= 0:
            base_recommendation = "SELL"
            base_confidence = 0.8  # High confidence for financially distressed companies
        else:
            # Calculate price to fair value ratio for positive fair values
            price_to_fair_value = current_price / fair_value
            
            # Base recommendation on price vs fair value
            if price_to_fair_value <= 0.85:  # Stock is undervalued by 15% or more
                base_recommendation = "BUY"
                base_confidence = 0.7
            elif price_to_fair_value >= 1.15:  # Stock is overvalued by 15% or more
                base_recommendation = "SELL"
                base_confidence = 0.7
            else:
                base_recommendation = "HOLD"
                base_confidence = 0.6
        
        # Adjust confidence based on PEG ratio
        peg_adjustment = 0.0
        if peg_ratio is not None:
            if peg_ratio < 1.0 and base_recommendation == "BUY":
                peg_adjustment = 0.1  # PEG < 1 supports buy recommendation
            elif peg_ratio > 2.0 and base_recommendation == "SELL":
                peg_adjustment = 0.1  # PEG > 2 supports sell recommendation
            elif 1.0 <= peg_ratio <= 1.5:
                peg_adjustment = 0.05  # Fair PEG ratio
        
        # Adjust confidence based on number of models used
        model_adjustment = min(0.1, num_models * 0.05)
        
        # Calculate final confidence
        final_confidence = min(1.0, base_confidence + peg_adjustment + model_adjustment)
        
        return base_recommendation, final_confidence
    
    def calculate_relative_valuation_metrics(
        self,
        symbol: str,
        stock_info: StockInfo,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        """Calculate additional relative valuation metrics.
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Basic stock information
            income_statement: Income statement data
            balance_sheet: Balance sheet data
            
        Returns:
            Dictionary of relative valuation metrics
        """
        logger.debug(f"Calculating relative valuation metrics for {symbol}")
        
        metrics = {}
        
        try:
            # Price-to-Book ratio (already in stock_info, but calculate for verification)
            if not balance_sheet.empty:
                latest_period = balance_sheet.columns[0]
                book_value_rows = [
                    'Total Stockholder Equity',
                    'Total Shareholders Equity',
                    'Stockholders Equity'
                ]
                
                book_value = None
                for row_name in book_value_rows:
                    if row_name in balance_sheet.index:
                        book_value = balance_sheet.loc[row_name, latest_period]
                        if pd.notna(book_value):
                            break
                
                if book_value is not None and book_value > 0:
                    shares_outstanding = self._get_shares_outstanding(balance_sheet, stock_info)
                    if shares_outstanding and shares_outstanding > 0:
                        book_value_per_share = book_value / shares_outstanding
                        pb_ratio = stock_info.current_price / book_value_per_share
                        metrics['pb_ratio'] = pb_ratio
            
            # Price-to-Sales ratio
            if not income_statement.empty:
                latest_period = income_statement.columns[0]
                revenue_rows = [
                    'Total Revenue',
                    'Revenue',
                    'Net Sales'
                ]
                
                revenue = None
                for row_name in revenue_rows:
                    if row_name in income_statement.index:
                        revenue = income_statement.loc[row_name, latest_period]
                        if pd.notna(revenue):
                            break
                
                if revenue is not None and revenue > 0:
                    ps_ratio = stock_info.market_cap / revenue
                    metrics['ps_ratio'] = ps_ratio
            
            # Enterprise Value to EBITDA (simplified)
            if not income_statement.empty:
                latest_period = income_statement.columns[0]
                
                # Try to find EBITDA or calculate it
                ebitda_rows = [
                    'EBITDA',
                    'Normalized EBITDA'
                ]
                
                ebitda = None
                for row_name in ebitda_rows:
                    if row_name in income_statement.index:
                        ebitda = income_statement.loc[row_name, latest_period]
                        if pd.notna(ebitda):
                            break
                
                # If EBITDA not found, approximate from operating income
                if ebitda is None:
                    operating_income = None
                    if 'Operating Income' in income_statement.index:
                        operating_income = income_statement.loc['Operating Income', latest_period]
                    
                    if operating_income is not None and pd.notna(operating_income):
                        # Rough approximation: assume D&A is 3-5% of revenue
                        revenue = None
                        for row_name in ['Total Revenue', 'Revenue']:
                            if row_name in income_statement.index:
                                revenue = income_statement.loc[row_name, latest_period]
                                if pd.notna(revenue):
                                    break
                        
                        if revenue is not None:
                            estimated_da = revenue * 0.04  # 4% approximation
                            ebitda = operating_income + estimated_da
                
                if ebitda is not None and ebitda > 0:
                    # Simplified EV calculation (Market Cap as proxy for EV)
                    ev_ebitda = stock_info.market_cap / ebitda
                    metrics['ev_ebitda'] = ev_ebitda
            
            # PEG ratio (already calculated in main method)
            peg_ratio = self.calculate_peg_ratio(symbol, stock_info, income_statement)
            if peg_ratio is not None:
                metrics['peg_ratio'] = peg_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating relative valuation metrics for {symbol}: {str(e)}")
        
        return metrics