# Power BI Integration Guide for Stock Analysis Dashboard

## Overview
This guide provides detailed instructions for setting up Power BI dashboards with the stock analysis data exported from the system. It covers data connection setup, template usage, and best practices for creating effective visualizations.

## Exported Data Formats
The stock analysis system exports data in three formats, all optimized for Power BI:

1. **CSV Format**: Single flattened file with all data points
2. **Excel Format**: Multiple sheets organized by data category
3. **JSON Format**: Structured format with metadata and schema definitions

## Setting Up Data Connections

### CSV Connection
1. Open Power BI Desktop
2. Click "Get Data" > "Text/CSV"
3. Browse to your exported CSV file (e.g., `stock_analysis_YYYYMMDD_HHMMSS.csv`)
4. In the preview dialog, ensure data types are correctly detected
5. Click "Load" to import the data

### Excel Connection
1. Open Power BI Desktop
2. Click "Get Data" > "Excel"
3. Browse to your exported Excel file (e.g., `stock_analysis_YYYYMMDD_HHMMSS.xlsx`)
4. In the Navigator dialog, select the sheets you want to import
   - For a complete dashboard, select all sheets
   - For specific analyses, select relevant sheets (e.g., Summary + Financial_Ratios)
5. Click "Load" to import the data

### JSON Connection
1. Open Power BI Desktop
2. Click "Get Data" > "JSON"
3. Browse to your exported JSON file (e.g., `stock_analysis_YYYYMMDD_HHMMSS.json`)
4. In the preview dialog, expand the nested structure to access the data sections
5. Click "Transform Data" to open Power Query Editor
6. Expand the nested JSON structure to create proper tables
7. Click "Close & Apply" to load the data

## Data Model Setup

### Creating Relationships
After loading data, set up relationships between tables:

1. Go to "Model" view in Power BI Desktop
2. Create relationships between tables using Symbol and Timestamp as keys:
   - Connect Summary to Stock_Info (Symbol, Timestamp)
   - Connect Summary to Financial_Ratios (Symbol, Timestamp)
   - Connect Summary to Health_Scores (Symbol, Timestamp)
   - Connect Summary to Valuation (Symbol, Timestamp)
   - Connect Summary to Sentiment (Symbol, Timestamp)
   - Connect Summary to Recommendations (Symbol, Timestamp)

### Creating Calculated Columns and Measures
Add these useful calculations to enhance your analysis:

```
# Price to Fair Value Ratio
Price to Fair Value = 
[Current_Price] / [Average_Fair_Value]

# Valuation Gap Percentage
Valuation Gap % = 
([Average_Fair_Value] - [Current_Price]) / [Current_Price]

# Health Score Category
Health Score Category = 
IF([Overall_Health_Score] >= 80, "Excellent", 
   IF([Overall_Health_Score] >= 60, "Good", 
      IF([Overall_Health_Score] >= 40, "Average", 
         IF([Overall_Health_Score] >= 20, "Poor", "Critical"))))

# Sentiment Category
Sentiment Category = 
IF([Overall_Sentiment] >= 0.3, "Positive", 
   IF([Overall_Sentiment] <= -0.3, "Negative", "Neutral"))

# Investment Score
Investment Score = 
VAR HealthWeight = 0.3
VAR ValuationWeight = 0.4
VAR SentimentWeight = 0.3
VAR NormalizedHealth = [Overall_Health_Score] / 100
VAR NormalizedValuation = IF([Current_Price] < [Average_Fair_Value], 
                            MIN(([Average_Fair_Value] - [Current_Price]) / [Current_Price], 1), 
                            0)
VAR NormalizedSentiment = ([Overall_Sentiment] + 1) / 2
RETURN (NormalizedHealth * HealthWeight) + 
       (NormalizedValuation * ValuationWeight) + 
       (NormalizedSentiment * SentimentWeight)
```

## Using the Template Files

### Template Installation
1. Download the template (.pbit) files from the powerbi_templates directory
2. Double-click the template file to open it in Power BI Desktop
3. When prompted, provide the path to your exported data file
4. The template will load with pre-configured visualizations

### Available Templates
1. **Financial Analysis Template**: Focus on financial ratios and health scores
2. **Valuation Analysis Template**: Focus on fair value calculations and investment recommendations
3. **Sentiment Analysis Template**: Focus on news sentiment and market perception

### Customizing Templates
1. Add or remove visualizations based on your specific needs
2. Modify color schemes and formatting to match your preferences
3. Add additional calculated measures for custom metrics
4. Adjust filters and slicers for your analysis focus

## Best Practices for Effective Dashboards

### Visual Design
1. Use consistent color coding across visualizations
   - Green for positive indicators (BUY, High Health Score)
   - Red for negative indicators (SELL, Low Health Score)
   - Yellow/Orange for neutral indicators (HOLD, Medium Health Score)
2. Organize related visualizations together
3. Include a title, description, and last updated timestamp
4. Use tooltips to provide additional context and details

### Performance Optimization
1. Limit the number of visualizations per page (5-7 maximum)
2. Use bookmarks for different analysis perspectives
3. Implement drillthrough pages for detailed analysis
4. Use filters to limit data shown in visuals

### Data Refresh
1. For local files, set up automatic refresh schedule
2. For Power BI Service, configure scheduled refresh
3. Consider incremental refresh for large datasets
4. Update the data source when new exports are available

## Testing Power BI Compatibility

To ensure your exported data works well with Power BI:

1. Test loading each export format (CSV, Excel, JSON)
2. Verify that data types are correctly interpreted
3. Check that relationships can be established between tables
4. Confirm that all required fields are available for visualizations
5. Test calculated measures with sample data
6. Verify that filters and slicers work as expected

## Troubleshooting Common Issues

### Data Type Issues
- Problem: Power BI incorrectly interprets data types
- Solution: Use Power Query Editor to explicitly set data types

### Relationship Issues
- Problem: Cannot create relationships between tables
- Solution: Ensure key columns have the same data type and no blank values

### Visualization Issues
- Problem: Visualizations show unexpected results
- Solution: Check data model relationships and measure calculations

### Performance Issues
- Problem: Dashboard is slow to load or interact with
- Solution: Reduce the number of visualizations, optimize measures, use filters

## Additional Resources
- [Power BI Documentation](https://docs.microsoft.com/en-us/power-bi/)
- [DAX Formula Reference](https://docs.microsoft.com/en-us/dax/dax-function-reference)
- [Power BI Community](https://community.powerbi.com/)
- [Power BI YouTube Channel](https://www.youtube.com/c/MSPowerBI)