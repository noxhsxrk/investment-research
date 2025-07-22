# Financial Analysis Dashboard Template

## Overview
This template provides comprehensive financial analysis visualizations for stock data, focusing on financial ratios, health scores, and performance metrics.

## Data Connection Setup
1. Open Power BI Desktop
2. Click "Get Data" > "Text/CSV" or "Excel" depending on your export format
3. Browse to your exported data file (e.g., `stock_analysis_YYYYMMDD_HHMMSS.csv` or `.xlsx`)
4. For Excel files, select all sheets when prompted
5. Click "Load" to import the data

## Recommended Visualizations

### Financial Health Overview
- **Card Visuals**:
  - Overall Health Score
  - Financial Strength
  - Risk Assessment
  - Current Price vs Fair Value
  
- **Gauge Visual**:
  - Overall Health Score (0-100 scale)
  - Financial Strength (0-100 scale)
  - Profitability Health (0-100 scale)
  - Liquidity Health (0-100 scale)

- **Conditional Formatting**:
  - Health scores: Green (>70), Yellow (40-70), Red (<40)
  - Risk Assessment: Green (Low), Yellow (Medium), Red (High)

### Financial Ratios Analysis
- **Matrix Visual**:
  - Rows: Symbol, Company Name
  - Values: Current Ratio, Quick Ratio, Debt-to-Equity, Return on Equity, Gross Margin
  - Conditional formatting based on industry benchmarks

- **Column Chart**:
  - X-axis: Symbol
  - Y-axis: Key financial ratios (Current Ratio, Quick Ratio, Debt-to-Equity)
  - Include average line for comparison

- **Radar Chart**:
  - Categories: Liquidity, Profitability, Leverage, Efficiency
  - Values: Normalized scores for each category
  - Multiple stocks for comparison

### Ratio Comparison
- **Scatter Plot**:
  - X-axis: Return on Equity
  - Y-axis: Debt-to-Equity
  - Size: Market Cap
  - Color: Sector
  - Tooltip: Company Name, Health Score

- **Table Visual**:
  - Columns: Symbol, Company Name, Current Ratio, Quick Ratio, Cash Ratio
  - Sorting: Descending by Overall Health Score
  - Conditional formatting for ratio values

## Data Model Relationships
- Connect Summary table to Financial Ratios using Symbol as the key
- Connect Summary table to Health Scores using Symbol as the key
- Create calculated columns for ratio comparisons and percentage differences

## Calculated Measures
```
# Liquidity Score
Liquidity Score = 
VAR CurrentRatioScore = IF([Current_Ratio] >= 2, 100, IF([Current_Ratio] >= 1, [Current_Ratio] * 50, [Current_Ratio] * 25))
VAR QuickRatioScore = IF([Quick_Ratio] >= 1, 100, [Quick_Ratio] * 100)
RETURN (CurrentRatioScore + QuickRatioScore) / 2

# Valuation Gap
Valuation Gap = 
([Average_Fair_Value] - [Current_Price]) / [Current_Price]

# Valuation Gap Percentage
Valuation Gap % = 
FORMAT(([Average_Fair_Value] - [Current_Price]) / [Current_Price], "0.00%")

# Risk-Adjusted Return Potential
Risk-Adjusted Return = 
VAR ReturnPotential = MAX(([Average_Fair_Value] - [Current_Price]) / [Current_Price], 0)
VAR RiskFactor = SWITCH([Risk_Assessment], "Low", 1, "Medium", 0.7, "High", 0.4)
RETURN ReturnPotential * RiskFactor
```

## Filters and Slicers
- Sector/Industry filter
- Health Score range slider
- Risk Assessment multi-select
- Valuation Recommendation filter

## Dashboard Layout
1. **Top Section**: Key metrics cards and gauges
2. **Middle Section**: Financial ratios comparison charts
3. **Bottom Section**: Detailed ratio tables with conditional formatting
4. **Right Panel**: Filters and slicers

## Refresh Settings
- Configure automatic refresh based on your data update frequency
- Set up incremental refresh if using Power BI Service