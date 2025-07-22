# Valuation Analysis Dashboard Template

## Overview
This template focuses on stock valuation metrics, fair value calculations, and investment recommendations based on valuation models.

## Data Connection Setup
1. Open Power BI Desktop
2. Click "Get Data" > "Text/CSV" or "Excel" depending on your export format
3. Browse to your exported data file (e.g., `stock_analysis_YYYYMMDD_HHMMSS.csv` or `.xlsx`)
4. For Excel files, select the Summary and Valuation sheets
5. Click "Load" to import the data

## Recommended Visualizations

### Valuation Overview
- **Card Visuals**:
  - Current Price
  - Average Fair Value
  - Valuation Gap (%)
  - Recommendation
  
- **Gauge Visual**:
  - Price to Fair Value Ratio (0.5-1.5 scale)
  - Confidence Level (0-1 scale)

- **Conditional Formatting**:
  - Valuation Gap: Green (>20%), Yellow (5-20%), Red (<5%)
  - Recommendation: Green (BUY), Yellow (HOLD), Red (SELL)

### Valuation Models Comparison
- **Column Chart**:
  - X-axis: Symbol
  - Y-axis: Current Price, DCF Value, Peer Comparison Value, Average Fair Value
  - Include data labels for values

- **Waterfall Chart**:
  - Show breakdown of valuation components
  - Start with Current Price
  - Show adjustments from different valuation models
  - End with Average Fair Value

- **Scatter Plot**:
  - X-axis: Market Cap
  - Y-axis: Price to Fair Value Ratio
  - Size: Confidence Level
  - Color: Recommendation
  - Tooltip: Company Name, Sector, Current Price, Fair Value

### Valuation Details
- **Table Visual**:
  - Columns: Symbol, Company Name, Current Price, DCF Value, Peer Comparison Value, Average Fair Value, Recommendation
  - Sorting: Descending by Valuation Gap
  - Conditional formatting for recommendation

- **Multi-row Card**:
  - Symbol and Company Name
  - Current Price vs Fair Value
  - Confidence Level
  - Recommendation with reasoning

## Data Model Relationships
- Connect Summary table to Valuation using Symbol as the key
- Create calculated columns for valuation gaps and percentage differences

## Calculated Measures
```
# Valuation Gap
Valuation Gap = 
[Average_Fair_Value] - [Current_Price]

# Valuation Gap Percentage
Valuation Gap % = 
FORMAT(([Average_Fair_Value] - [Current_Price]) / [Current_Price], "0.00%")

# Price to Fair Value Ratio
P/FV Ratio = 
[Current_Price] / [Average_Fair_Value]

# Investment Attractiveness Score
Investment Score = 
VAR ValuationScore = IF([Current_Price] < [Average_Fair_Value], 
                        MIN(([Average_Fair_Value] - [Current_Price]) / [Current_Price] * 100, 100), 
                        0)
VAR ConfidenceAdjustment = [Confidence_Level] * 100
RETURN (ValuationScore * 0.7) + (ConfidenceAdjustment * 0.3)
```

## Filters and Slicers
- Sector/Industry filter
- Recommendation filter (BUY/HOLD/SELL)
- Confidence Level range slider
- Valuation Gap percentage range

## Dashboard Layout
1. **Top Section**: Key valuation metrics and recommendation cards
2. **Middle Section**: Valuation models comparison charts
3. **Bottom Section**: Detailed valuation table with conditional formatting
4. **Right Panel**: Filters and slicers

## Refresh Settings
- Configure automatic refresh based on your data update frequency
- Set up incremental refresh if using Power BI Service