# Sentiment Analysis Dashboard Template

## Overview
This template focuses on news sentiment analysis, key themes, and sentiment trends for stocks to provide insights into market perception.

## Data Connection Setup
1. Open Power BI Desktop
2. Click "Get Data" > "Text/CSV" or "Excel" depending on your export format
3. Browse to your exported data file (e.g., `stock_analysis_YYYYMMDD_HHMMSS.csv` or `.xlsx`)
4. For Excel files, select the Summary and Sentiment sheets
5. Click "Load" to import the data

## Recommended Visualizations

### Sentiment Overview
- **Card Visuals**:
  - Overall Sentiment Score
  - Total Articles Analyzed
  - Positive to Negative Ratio
  - Key Themes Count
  
- **Gauge Visual**:
  - Overall Sentiment (-1 to 1 scale)
  - Sentiment Trend Direction

- **Conditional Formatting**:
  - Sentiment Score: Green (>0.3), Yellow (-0.3 to 0.3), Red (<-0.3)
  - Positive/Negative Ratio: Green (>1.5), Yellow (0.8-1.5), Red (<0.8)

### Sentiment Distribution
- **Donut Chart**:
  - Categories: Positive, Negative, Neutral
  - Values: Count of articles in each category
  - Include percentage labels

- **Column Chart**:
  - X-axis: Symbol
  - Y-axis: Positive Count, Negative Count, Neutral Count (stacked)
  - Include data labels for values

- **Scatter Plot**:
  - X-axis: Overall Sentiment
  - Y-axis: Total Articles
  - Size: Market Cap
  - Color: Sector
  - Tooltip: Company Name, Key Themes

### Sentiment Trends and Themes
- **Line Chart**:
  - X-axis: Time periods
  - Y-axis: Sentiment Trend values
  - Multiple stocks for comparison

- **Word Cloud**:
  - Words: Key Themes
  - Size: Frequency of theme appearance
  - Color: Sentiment association (positive/negative/neutral)

- **Table Visual**:
  - Columns: Symbol, Company Name, Overall Sentiment, Positive Count, Negative Count, Neutral Count, Key Themes
  - Sorting: Descending by Overall Sentiment
  - Conditional formatting for sentiment values

## Data Model Relationships
- Connect Summary table to Sentiment using Symbol as the key
- Create calculated columns for sentiment analysis and article distributions

## Calculated Measures
```
# Positive to Negative Ratio
Pos/Neg Ratio = 
IF([Negative_Count] = 0, [Positive_Count], [Positive_Count] / [Negative_Count])

# Sentiment Strength
Sentiment Strength = 
ABS([Overall_Sentiment])

# Sentiment Direction
Sentiment Direction = 
IF([Overall_Sentiment] > 0, "Positive", IF([Overall_Sentiment] < 0, "Negative", "Neutral"))

# Sentiment Trend Direction
Sentiment Trend Direction = 
VAR LastValue = LAST([Sentiment_Trend])
VAR PreviousValue = LASTNONBLANK(EARLIER([Sentiment_Trend]), 1)
RETURN IF(LastValue > PreviousValue, "Improving", IF(LastValue < PreviousValue, "Declining", "Stable"))

# Article Distribution
Article Distribution = 
VAR Total = [Positive_Count] + [Negative_Count] + [Neutral_Count]
RETURN 
    "Positive: " & FORMAT([Positive_Count] / Total, "0.0%") & 
    ", Negative: " & FORMAT([Negative_Count] / Total, "0.0%") & 
    ", Neutral: " & FORMAT([Neutral_Count] / Total, "0.0%")
```

## Filters and Slicers
- Sector/Industry filter
- Sentiment range slider
- Key Themes multi-select
- Article count minimum threshold

## Dashboard Layout
1. **Top Section**: Key sentiment metrics and gauge
2. **Middle Section**: Sentiment distribution and trend charts
3. **Bottom Section**: Key themes word cloud and detailed sentiment table
4. **Right Panel**: Filters and slicers

## Refresh Settings
- Configure automatic refresh based on your data update frequency
- Set up incremental refresh if using Power BI Service