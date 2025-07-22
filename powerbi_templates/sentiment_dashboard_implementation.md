# Sentiment Analysis Dashboard Implementation Guide

This guide provides step-by-step instructions for implementing the Sentiment Analysis Dashboard in Power BI Desktop using the sample data provided.

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
   - Overall_Sentiment: Decimal Number
   - Positive_News_Count, Negative_News_Count, Neutral_News_Count: Whole Number
   - Key_Themes, Sentiment_Trend: Text
5. Click "Load" to import the data

## Step 2: Prepare the Data

1. Click on "Transform Data" to open Power Query Editor
2. Split the Key_Themes column:
   - Select the Key_Themes column
   - Click "Split Column" > "By Delimiter"
   - Choose semicolon as the delimiter
   - Split into columns
   - Rename the resulting columns as "Theme1", "Theme2", etc.

3. Split the Sentiment_Trend column:
   - Select the Sentiment_Trend column
   - Click "Split Column" > "By Delimiter"
   - Choose semicolon as the delimiter
   - Split into columns
   - Rename the resulting columns as "Trend1", "Trend2", etc.
   - Change the data type to Decimal Number

4. Create a new column for Total Articles:
   ```
   Total Articles = [Positive_News_Count] + [Negative_News_Count] + [Neutral_News_Count]
   ```

5. Click "Close & Apply" to apply the transformations

## Step 3: Create Calculated Columns and Measures

1. Click on the "Model" view in the ribbon
2. Right-click on the table and select "New Column" or "New Measure"
3. Create the following calculated columns and measures:

```
# Positive to Negative Ratio
Pos/Neg Ratio = 
IF([Negative_News_Count] = 0, [Positive_News_Count], [Positive_News_Count] / [Negative_News_Count])

# Sentiment Strength
Sentiment Strength = 
ABS([Overall_Sentiment])

# Sentiment Direction
Sentiment Direction = 
IF([Overall_Sentiment] > 0, "Positive", IF([Overall_Sentiment] < 0, "Negative", "Neutral"))

# Sentiment Trend Direction
Sentiment Trend Direction = 
VAR LastValue = [Trend5]
VAR FirstValue = [Trend1]
RETURN IF(LastValue > FirstValue, "Improving", IF(LastValue < FirstValue, "Declining", "Stable"))

# Positive Article Percentage
Positive Article % = 
DIVIDE([Positive_News_Count], [Positive_News_Count] + [Negative_News_Count] + [Neutral_News_Count])

# Negative Article Percentage
Negative Article % = 
DIVIDE([Negative_News_Count], [Positive_News_Count] + [Negative_News_Count] + [Neutral_News_Count])

# Neutral Article Percentage
Neutral Article % = 
DIVIDE([Neutral_News_Count], [Positive_News_Count] + [Negative_News_Count] + [Neutral_News_Count])
```

## Step 4: Create the Sentiment Overview Section

1. Return to the "Report" view
2. Add a new page titled "Sentiment Analysis"
3. Add the following card visuals:

### Card Visuals
1. **Overall Sentiment Score Card**:
   - Drag "Overall_Sentiment" to a card visual
   - Format: Two decimal places
   - Title: "Overall Sentiment Score"
   - Conditional formatting: 
     - Rules: >0.3 (Green), -0.3 to 0.3 (Yellow), <-0.3 (Red)

2. **Total Articles Card**:
   - Drag "Total Articles" to a card visual
   - Format: No decimal places
   - Title: "Total Articles Analyzed"

3. **Positive to Negative Ratio Card**:
   - Drag "Pos/Neg Ratio" to a card visual
   - Format: One decimal place
   - Title: "Positive to Negative Ratio"
   - Conditional formatting: 
     - Rules: >1.5 (Green), 0.8-1.5 (Yellow), <0.8 (Red)

4. **Sentiment Direction Card**:
   - Drag "Sentiment Direction" to a card visual
   - Title: "Sentiment Direction"
   - Conditional formatting:
     - Rules: "Positive" (Green), "Neutral" (Yellow), "Negative" (Red)

### Gauge Visual
1. **Overall Sentiment Gauge**:
   - Create a gauge visual
   - Value: Overall_Sentiment
   - Minimum: -1
   - Maximum: 1
   - Target: 0.5
   - Format with data labels and title "Overall Sentiment"
   - Conditional formatting:
     - >0.3 (Green), -0.3 to 0.3 (Yellow), <-0.3 (Red)

## Step 5: Create Sentiment Distribution Section

### Donut Chart
1. Create a donut chart visual
2. Add a calculated table for sentiment distribution:
   ```
   Sentiment Distribution = 
   SUMMARIZE(
       'Table',
       "Category", "Positive",
       "Count", SUM([Positive_News_Count])
   )
   UNION
   SUMMARIZE(
       'Table',
       "Category", "Negative",
       "Count", SUM([Negative_News_Count])
   )
   UNION
   SUMMARIZE(
       'Table',
       "Category", "Neutral",
       "Count", SUM([Neutral_News_Count])
   )
   ```
3. Add "Category" to the Legend field
4. Add "Count" to the Values field
5. Format the chart:
   - Title: "Sentiment Distribution"
   - Data labels: On, show percentage
   - Colors: Positive (Green), Neutral (Yellow), Negative (Red)

### Column Chart
1. Create a column chart visual
2. Add "Symbol" to the Axis field
3. Add the following to the Values field:
   - Positive_News_Count
   - Negative_News_Count
   - Neutral_News_Count
4. Format the chart:
   - Title: "Article Count by Sentiment"
   - Data labels: On
   - Legend: On, positioned at the bottom
   - Y-axis: Start at 0, title "Number of Articles"
   - Colors: Positive_News_Count (Green), Neutral_News_Count (Yellow), Negative_News_Count (Red)

### Scatter Plot
1. Create a scatter chart visual
2. Add "Overall_Sentiment" to the X-axis field
3. Add "Total Articles" to the Y-axis field
4. Add "Market_Cap" to the Size field
5. Add "Sector" to the Legend field
6. Add the following to the Tooltip field:
   - Company_Name
   - Key_Themes
7. Format the chart:
   - Title: "Sentiment vs Coverage"
   - Data labels: Off
   - Legend: On, positioned at the bottom
   - Add reference line at X=0

## Step 6: Create Sentiment Trends and Themes Section

### Line Chart
1. Create a line chart visual
2. Add a calculated table for sentiment trends:
   ```
   Sentiment Trends = 
   SELECTCOLUMNS(
       'Table',
       "Symbol", [Symbol],
       "Period 1", [Trend1],
       "Period 2", [Trend2],
       "Period 3", [Trend3],
       "Period 4", [Trend4],
       "Period 5", [Trend5]
   )
   ```
3. Add "Symbol" to the Legend field
4. Add "Period 1" through "Period 5" to the Values field
5. Format the chart:
   - Title: "Sentiment Trends"
   - Data labels: Off
   - Legend: On, positioned at the bottom
   - Y-axis: Range from -1 to 1, title "Sentiment Score"

### Word Cloud (if available in your Power BI version)
1. If using Power BI Desktop with custom visuals:
   - Import the Word Cloud custom visual from the marketplace
   - Create a word cloud visual
   - Add "Theme1", "Theme2", etc. to the Category field
   - Format the word cloud:
     - Title: "Key Themes"
     - Colors: Use a gradient based on sentiment

### Table Visual
1. Create a table visual
2. Add the following fields:
   - Symbol
   - Company_Name
   - Overall_Sentiment
   - Positive_News_Count
   - Negative_News_Count
   - Neutral_News_Count
   - Theme1
   - Theme2
   - Theme3
3. Format the table:
   - Title: "Sentiment Details"
   - Sort by: Overall_Sentiment (descending)
   - Conditional formatting for Overall_Sentiment:
     - >0.3 (Green), -0.3 to 0.3 (Yellow), <-0.3 (Red)

## Step 7: Add Filters and Slicers

1. **Sector/Industry Filter**:
   - Create a slicer visual
   - Add "Sector" to the Field
   - Style: Dropdown
   - Title: "Filter by Sector"

2. **Sentiment Range Slider**:
   - Create a slicer visual
   - Add "Overall_Sentiment" to the Field
   - Style: Between
   - Title: "Sentiment Range"

3. **Key Themes Filter**:
   - Create a slicer visual
   - Add "Theme1" to the Field
   - Style: List
   - Title: "Filter by Key Theme"

4. **Article Count Minimum**:
   - Create a slicer visual
   - Add "Total Articles" to the Field
   - Style: Greater than or equal to
   - Title: "Minimum Article Count"

## Step 8: Finalize Dashboard Layout

1. Arrange the visuals in the following layout:
   - **Top Section**: Card visuals and gauge for key sentiment metrics
   - **Middle Section**: Sentiment distribution and trend charts
   - **Bottom Section**: Key themes word cloud and detailed sentiment table
   - **Right Panel**: Filters and slicers

2. Add a title to the page: "News Sentiment Analysis Dashboard"

3. Add a subtitle with dynamic timestamp: "Data as of [Timestamp]"

4. Add navigation buttons to other dashboard pages (if applicable)

5. Apply a consistent theme to the dashboard:
   - Go to the "View" tab
   - Click "Themes"
   - Select a professional theme or create a custom one

## Step 9: Set Up Refresh Settings

1. Go to "File" > "Options and settings" > "Data source settings"
2. Select your data source and click "Edit permissions"
3. Configure the appropriate refresh settings
4. If publishing to Power BI Service, set up scheduled refresh

## Step 10: Save and Export the Template

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
|              News Sentiment Analysis Dashboard         |
+---------------+-------------------+-------------------+
| Sentiment     | Total Articles    | Pos/Neg Ratio     |
| 0.30          | 30                | 3.0               |
+---------------+-------------------+-------------------+
| [Gauge: 0.3 on -1 to 1 scale]     | Sentiment: Positive |
+-------------------------------------------------------+
|                         |                             |
|  [Donut: Sentiment      |  [Column: Article Count     |
|   Distribution]         |   by Sentiment]             |
|                         |                             |
+-------------------------------------------------------+
|                                                       |
|  [Line Chart: Sentiment Trends]                       |
|                                                       |
+-------------------------------------------------------+
|                         |                             |
|  [Word Cloud:           |  [Scatter: Sentiment        |
|   Key Themes]           |   vs Coverage]              |
|                         |                             |
+-------------------------------------------------------+
|                                                       |
|  [Table: Sentiment Details]                           |
|                                                       |
+-------------------------------------------------------+
```

This implementation guide provides a detailed walkthrough for creating a professional news sentiment analysis dashboard in Power BI. By following these steps, you'll create a powerful tool for analyzing market sentiment and making informed investment decisions based on news coverage.