# Financial Analysis Dashboard Implementation Guide

This guide provides step-by-step instructions for implementing the Financial Analysis Dashboard in Power BI Desktop using the sample data provided.

## Prerequisites
- Power BI Desktop installed (latest version recommended)
- Sample data file: `powerbi_templates/sample_data/stock_analysis_sample.csv`

## Step 1: Create a New Power BI Project and Import Data

1. Open Power BI Desktop
2. Select "Get Data" > "Text/CSV"
3. Browse to the `stock_analysis_sample.csv` file and select it
4. In the preview dialog, ensure data types are correctly detected:
   - Symbol: Text
   - Timestamp: Date/Time
   - Current_Price, Market_Cap, etc.: Decimal Number
5. Click "Load" to import the data

## Step 2: Create Calculated Columns and Measures

1. Click on the "Model" view in the ribbon
2. Right-click on the table and select "New Column" or "New Measure"
3. Create the following calculated columns and measures:

```
# Liquidity Score
Liquidity Score = 
VAR CurrentRatioScore = IF([Current_Ratio] >= 2, 100, IF([Current_Ratio] >= 1, [Current_Ratio] * 50, [Current_Ratio] * 25))
VAR QuickRatioScore = IF([Quick_Ratio] >= 1, 100, [Quick_Ratio] * 100)
RETURN (CurrentRatioScore + QuickRatioScore) / 2

# Profitability Score
Profitability Score = 
VAR GrossMarginScore = [Gross_Margin] * 100
VAR NetMarginScore = [Net_Profit_Margin] * 200
VAR ROEScore = [Return_On_Equity] * 100
RETURN (GrossMarginScore + NetMarginScore + ROEScore) / 3

# Leverage Risk
Leverage Risk = 
IF([Debt_To_Equity] > 2, "High", IF([Debt_To_Equity] > 1, "Medium", "Low"))

# Health Score Category
Health Score Category = 
IF([Overall_Health_Score] >= 80, "Excellent", 
   IF([Overall_Health_Score] >= 60, "Good", 
      IF([Overall_Health_Score] >= 40, "Average", 
         IF([Overall_Health_Score] >= 20, "Poor", "Critical"))))
```

## Step 3: Create the Financial Health Overview Section

1. Return to the "Report" view
2. Add a new page titled "Financial Health Overview"
3. Add the following card visuals:

### Card Visuals
1. **Overall Health Score Card**:
   - Drag "Overall_Health_Score" to a card visual
   - Format: No decimal places
   - Title: "Overall Health Score"
   - Conditional formatting: 
     - Rules: >70 (Green), 40-70 (Yellow), <40 (Red)

2. **Financial Strength Card**:
   - Drag "Financial_Strength" to a card visual
   - Format: No decimal places
   - Title: "Financial Strength"
   - Conditional formatting: 
     - Rules: >70 (Green), 40-70 (Yellow), <40 (Red)

3. **Risk Assessment Card**:
   - Drag "Risk_Assessment" to a card visual
   - Title: "Risk Assessment"
   - Conditional formatting:
     - Rules: "Low" (Green), "Medium" (Yellow), "High" (Red)

4. **Current Price vs Fair Value Card**:
   - Create a calculated measure: 
     ```
     Price vs Fair Value = [Current_Price] & " / " & [Average_Fair_Value]
     ```
   - Add this measure to a card visual
   - Title: "Current Price / Fair Value"

### Gauge Visuals
1. **Overall Health Score Gauge**:
   - Create a gauge visual
   - Value: Overall_Health_Score
   - Minimum: 0
   - Maximum: 100
   - Target: 80
   - Format with data labels and title "Overall Health Score"
   - Conditional formatting:
     - >70 (Green), 40-70 (Yellow), <40 (Red)

2. **Financial Strength Gauge**:
   - Create a gauge visual
   - Value: Financial_Strength
   - Minimum: 0
   - Maximum: 100
   - Target: 80
   - Format with data labels and title "Financial Strength"
   - Conditional formatting:
     - >70 (Green), 40-70 (Yellow), <40 (Red)

3. **Profitability Health Gauge**:
   - Create a gauge visual
   - Value: Profitability_Health
   - Minimum: 0
   - Maximum: 100
   - Target: 80
   - Format with data labels and title "Profitability Health"
   - Conditional formatting:
     - >70 (Green), 40-70 (Yellow), <40 (Red)

4. **Liquidity Health Gauge**:
   - Create a gauge visual
   - Value: Liquidity_Health
   - Minimum: 0
   - Maximum: 100
   - Target: 80
   - Format with data labels and title "Liquidity Health"
   - Conditional formatting:
     - >70 (Green), 40-70 (Yellow), <40 (Red)

## Step 4: Create Financial Ratios Analysis Section

### Matrix Visual
1. Create a matrix visual
2. Add "Symbol" to the Rows field
3. Add "Company_Name" to the Rows field (below Symbol)
4. Add the following to the Values field:
   - Current_Ratio
   - Quick_Ratio
   - Debt_To_Equity
   - Return_On_Equity
   - Gross_Margin
5. Format the matrix:
   - Title: "Key Financial Ratios"
   - Conditional formatting for each ratio:
     - Current_Ratio: >2 (Green), 1-2 (Yellow), <1 (Red)
     - Quick_Ratio: >1 (Green), 0.5-1 (Yellow), <0.5 (Red)
     - Debt_To_Equity: <1 (Green), 1-2 (Yellow), >2 (Red)
     - Return_On_Equity: >0.2 (Green), 0.1-0.2 (Yellow), <0.1 (Red)
     - Gross_Margin: >0.4 (Green), 0.2-0.4 (Yellow), <0.2 (Red)

### Column Chart
1. Create a column chart visual
2. Add "Symbol" to the Axis field
3. Add the following to the Values field:
   - Current_Ratio
   - Quick_Ratio
   - Debt_To_Equity
4. Add a line for industry average (if available) or create a calculated average
5. Format the chart:
   - Title: "Liquidity and Leverage Ratios"
   - Data labels: On
   - Legend: On, positioned at the bottom
   - Y-axis: Start at 0, title "Ratio Value"

### Radar Chart
1. Create a radar chart visual (if available in your Power BI version, otherwise use a line chart)
2. Add "Symbol" to the Legend field
3. Create four calculated measures for normalized scores:
   ```
   Normalized Liquidity = [Liquidity_Health] / 100
   Normalized Profitability = [Profitability_Health] / 100
   Normalized Leverage = 1 - ([Debt_To_Equity] / 3) // Capped at 3 for normalization
   Normalized Efficiency = [Asset_Turnover] / 2 // Capped at 2 for normalization
   ```
4. Add these normalized measures to the Values field
5. Format the chart:
   - Title: "Financial Balance Analysis"
   - Data labels: Off
   - Legend: On, positioned at the bottom

## Step 5: Create Ratio Comparison Section

### Scatter Plot
1. Create a scatter chart visual
2. Add "Return_On_Equity" to the X-axis field
3. Add "Debt_To_Equity" to the Y-axis field
4. Add "Market_Cap" to the Size field
5. Add "Sector" to the Legend field
6. Add the following to the Tooltip field:
   - Company_Name
   - Overall_Health_Score
   - Risk_Assessment
7. Format the chart:
   - Title: "Return on Equity vs Debt to Equity"
   - Data labels: Off
   - Legend: On, positioned at the bottom
   - Add reference lines at X=0.15 and Y=1.5

### Table Visual
1. Create a table visual
2. Add the following fields:
   - Symbol
   - Company_Name
   - Current_Ratio
   - Quick_Ratio
   - Cash_Ratio
   - Overall_Health_Score
3. Format the table:
   - Title: "Liquidity Ratios"
   - Sort by: Overall_Health_Score (descending)
   - Conditional formatting for ratio values:
     - Current_Ratio: >2 (Green), 1-2 (Yellow), <1 (Red)
     - Quick_Ratio: >1 (Green), 0.5-1 (Yellow), <0.5 (Red)
     - Cash_Ratio: >0.5 (Green), 0.2-0.5 (Yellow), <0.2 (Red)

## Step 6: Add Filters and Slicers

1. **Sector/Industry Filter**:
   - Create a slicer visual
   - Add "Sector" to the Field
   - Style: Dropdown
   - Title: "Filter by Sector"

2. **Health Score Range Slider**:
   - Create a slicer visual
   - Add "Overall_Health_Score" to the Field
   - Style: Between
   - Title: "Health Score Range"

3. **Risk Assessment Filter**:
   - Create a slicer visual
   - Add "Risk_Assessment" to the Field
   - Style: Buttons
   - Title: "Filter by Risk Level"

## Step 7: Finalize Dashboard Layout

1. Arrange the visuals in the following layout:
   - **Top Section**: Card visuals and gauges for key metrics
   - **Middle Section**: Financial ratios comparison charts
   - **Bottom Section**: Detailed ratio tables with conditional formatting
   - **Right Panel**: Filters and slicers

2. Add a title to the page: "Financial Health Analysis Dashboard"

3. Add a subtitle with dynamic timestamp: "Data as of [Timestamp]"

4. Add navigation buttons to other dashboard pages (if applicable)

5. Apply a consistent theme to the dashboard:
   - Go to the "View" tab
   - Click "Themes"
   - Select a professional theme or create a custom one

## Step 8: Set Up Refresh Settings

1. Go to "File" > "Options and settings" > "Data source settings"
2. Select your data source and click "Edit permissions"
3. Configure the appropriate refresh settings
4. If publishing to Power BI Service, set up scheduled refresh

## Step 9: Save and Export the Template

1. Save your Power BI file (.pbix)
2. To create a template:
   - Go to "File" > "Export" > "Power BI template"
   - Save as a .pbit file
   - This template can now be shared with others

## Final Checks

1. Test all visualizations to ensure they display correctly
2. Verify that filters and slicers work as expected
3. Check that conditional formatting is applied correctly
4. Ensure all data labels and titles are clear and informative
5. Test the dashboard with different data selections

## Example Dashboard Preview

When completed, your dashboard should look similar to this:

```
+-------------------------------------------------------+
|              Financial Health Analysis Dashboard       |
+---------------+-------------------+-------------------+
| Health Score  | Financial Strength| Risk Assessment   |
| 85            | 88                | Low               |
+---------------+-------------------+-------------------+
| [Gauge: 85/100] [Gauge: 88/100] [Gauge: 82/100] [Gauge: 90/100] |
+-------------------------------------------------------+
|                                                       |
|  [Matrix: Key Financial Ratios]                       |
|                                                       |
+-------------------------------------------------------+
|                                                       |
|  [Column Chart: Liquidity and Leverage Ratios]        |
|                                                       |
+-------------------------------------------------------+
|                                                       |
|  [Scatter Plot: Return on Equity vs Debt to Equity]   |
|                                                       |
+-------------------------------------------------------+
|                                                       |
|  [Table: Liquidity Ratios]                            |
|                                                       |
+-------------------------------------------------------+
```

This implementation guide provides a detailed walkthrough for creating a professional financial health analysis dashboard in Power BI. By following these steps, you'll create a powerful tool for analyzing company financial health and making informed investment decisions.