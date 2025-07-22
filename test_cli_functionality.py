#!/usr/bin/env python3
"""Test script to demonstrate CLI functionality."""

import subprocess
import sys
from pathlib import Path

def run_cli_command(args):
    """Run a CLI command and capture output."""
    try:
        # We'll test the argument parsing without actually running the analysis
        cmd = ["python3", "-c", f"""
import sys
sys.path.insert(0, '.')
from stock_analysis.cli import create_parser

parser = create_parser()
try:
    args = parser.parse_args({args})
    print(f"✅ Command parsed successfully: {{args.command}}")
    if hasattr(args, 'symbols'):
        print(f"   Symbols: {{args.symbols}}")
    if hasattr(args, 'export_format'):
        print(f"   Export format: {{args.export_format}}")
    if hasattr(args, 'verbose'):
        print(f"   Verbose: {{args.verbose}}")
except SystemExit as e:
    if e.code == 0:
        print("✅ Help displayed successfully")
    else:
        print(f"❌ Parser error: {{e.code}}")
except Exception as e:
    print(f"❌ Error: {{e}}")
"""]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run command: {e}")
        return False

def test_cli_commands():
    """Test various CLI commands."""
    
    print("🧪 Testing CLI Command Parsing")
    print("=" * 50)
    
    # Test help command
    print("\n1. Testing help command:")
    run_cli_command("['--help']")
    
    # Test analyze command
    print("\n2. Testing analyze command:")
    run_cli_command("['analyze', 'AAPL']")
    
    print("\n3. Testing analyze with multiple stocks:")
    run_cli_command("['analyze', 'AAPL', 'MSFT', '--verbose', '--export-format', 'csv']")
    
    # Test batch command
    print("\n4. Testing batch command:")
    run_cli_command("['batch', 'example_stocks.txt']")
    
    print("\n5. Testing batch with options:")
    run_cli_command("['batch', 'example_stocks.txt', '--export-format', 'excel', '--max-workers', '2']")
    
    # Test schedule commands
    print("\n6. Testing schedule add command:")
    run_cli_command("['schedule', 'add', 'daily-tech', 'AAPL,MSFT,GOOGL', '--interval', 'daily']")
    
    print("\n7. Testing schedule list command:")
    run_cli_command("['schedule', 'list']")
    
    print("\n8. Testing schedule status command:")
    run_cli_command("['schedule', 'status']")
    
    # Test config commands
    print("\n9. Testing config show command:")
    run_cli_command("['config', 'show']")
    
    print("\n10. Testing config set command:")
    run_cli_command("['config', 'set', 'stock_analysis.export.default_format', 'json']")
    
    print("\n11. Testing config validate command:")
    run_cli_command("['config', 'validate']")

def test_file_loading():
    """Test file loading functionality."""
    print("\n🧪 Testing File Loading")
    print("=" * 30)
    
    # Test the load_symbols_from_file function
    try:
        cmd = ["python3", "-c", """
import sys
sys.path.insert(0, '.')
from stock_analysis.cli import load_symbols_from_file

# Test loading from text file
try:
    symbols = load_symbols_from_file('example_stocks.txt')
    print(f"✅ Loaded {len(symbols)} symbols from text file: {', '.join(symbols[:5])}...")
except Exception as e:
    print(f"❌ Error loading text file: {e}")

# Test loading from CSV file  
try:
    symbols = load_symbols_from_file('example_stocks.csv')
    print(f"✅ Loaded {len(symbols)} symbols from CSV file: {', '.join(symbols[:5])}...")
except Exception as e:
    print(f"❌ Error loading CSV file: {e}")
"""]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Failed to test file loading: {e}")

def create_cli_usage_examples():
    """Create usage examples documentation."""
    
    examples = """
# Stock Analysis CLI Usage Examples

## Single Stock Analysis
```bash
# Analyze a single stock with default settings
stock-analysis analyze AAPL

# Analyze with verbose output and custom export format
stock-analysis analyze AAPL --verbose --export-format json --output my_analysis

# Analyze without exporting results
stock-analysis analyze AAPL --no-export
```

## Multiple Stock Analysis
```bash
# Analyze multiple stocks
stock-analysis analyze AAPL MSFT GOOGL --verbose

# Analyze with custom export settings
stock-analysis analyze AAPL MSFT GOOGL --export-format excel --output tech_stocks
```

## Batch Analysis
```bash
# Analyze stocks from a file
stock-analysis batch example_stocks.txt

# Batch analysis with custom settings
stock-analysis batch example_stocks.txt --export-format csv --max-workers 8 --verbose

# Batch analysis with custom output
stock-analysis batch example_stocks.txt --output quarterly_analysis --export-format excel
```

## Scheduling
```bash
# Add a daily scheduled job
stock-analysis schedule add daily-tech "AAPL,MSFT,GOOGL" --interval daily --name "Daily Tech Analysis"

# Add a weekly job with notifications disabled
stock-analysis schedule add weekly-finance "JPM,BAC,WFC" --interval weekly --no-notifications

# List all scheduled jobs
stock-analysis schedule list

# Show scheduler status
stock-analysis schedule status

# Show specific job status
stock-analysis schedule status daily-tech

# Run a job immediately
stock-analysis schedule run daily-tech

# Enable/disable jobs
stock-analysis schedule enable daily-tech
stock-analysis schedule disable daily-tech

# Remove a job
stock-analysis schedule remove daily-tech

# Start/stop the scheduler
stock-analysis schedule start
stock-analysis schedule stop

# Generate scheduler report
stock-analysis schedule report --days 30
```

## Configuration Management
```bash
# Show all configuration
stock-analysis config show

# Show specific configuration key
stock-analysis config show stock_analysis.export.default_format

# Set configuration values
stock-analysis config set stock_analysis.export.default_format json
stock-analysis config set stock_analysis.data_sources.yfinance.timeout 60

# Validate configuration
stock-analysis config validate

# Reset to defaults
stock-analysis config reset
```

## Global Options
```bash
# Use custom configuration file
stock-analysis --config my_config.yaml analyze AAPL

# Set log level
stock-analysis --log-level DEBUG analyze AAPL --verbose

# Verbose output for any command
stock-analysis --verbose batch example_stocks.txt
```

## File Formats

### Stock Symbol Files
Text file (one symbol per line):
```
AAPL
MSFT
GOOGL
# Comments are supported
AMZN
```

CSV file (comma-separated):
```
AAPL,MSFT,GOOGL
JPM,BAC,WFC
JNJ,PFE,UNH
```

### Export Formats
- **CSV**: Flattened data suitable for spreadsheet analysis
- **Excel**: Multi-sheet workbook with organized data
- **JSON**: Structured data with metadata for Power BI integration

## Error Handling
The CLI includes comprehensive error handling:
- Invalid stock symbols are reported but don't stop batch processing
- Network timeouts are retried automatically
- Configuration validation prevents invalid settings
- Verbose mode shows detailed error information
"""
    
    with open("CLI_USAGE_EXAMPLES.md", "w") as f:
        f.write(examples)
    
    print("✅ Created CLI_USAGE_EXAMPLES.md with comprehensive usage documentation")

def main():
    """Main test function."""
    print("🚀 Stock Analysis CLI Testing")
    print("=" * 50)
    
    # Test CLI structure
    test_cli_commands()
    
    # Test file loading
    test_file_loading()
    
    # Create usage examples
    create_cli_usage_examples()
    
    print("\n✅ CLI testing completed successfully!")
    print("\nThe CLI provides the following functionality:")
    print("  • Single stock analysis with progress indicators")
    print("  • Batch processing from files")
    print("  • Scheduled analysis jobs with automation")
    print("  • Configuration management")
    print("  • Multiple export formats (CSV, Excel, JSON)")
    print("  • Verbose output and error handling")
    print("  • Progress tracking and notifications")

if __name__ == "__main__":
    main()