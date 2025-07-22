#!/usr/bin/env python3
"""Test script to verify CLI structure without running it."""

import sys
import ast
import inspect
from pathlib import Path

def test_cli_structure():
    """Test the CLI module structure."""
    
    # Read the CLI file
    cli_path = Path("stock_analysis/cli.py")
    if not cli_path.exists():
        print("❌ CLI file not found")
        return False
    
    with open(cli_path, 'r') as f:
        content = f.read()
    
    # Parse the AST to check structure
    try:
        tree = ast.parse(content)
        print("✅ CLI file syntax is valid")
    except SyntaxError as e:
        print(f"❌ Syntax error in CLI file: {e}")
        return False
    
    # Check for required functions
    required_functions = [
        'create_parser',
        'handle_analyze_command',
        'handle_batch_command', 
        'handle_schedule_command',
        'handle_config_command',
        'main'
    ]
    
    function_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
    
    missing_functions = []
    for func in required_functions:
        if func in function_names:
            print(f"✅ Found function: {func}")
        else:
            print(f"❌ Missing function: {func}")
            missing_functions.append(func)
    
    # Check for required classes
    required_classes = ['ProgressIndicator']
    class_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
    
    for cls in required_classes:
        if cls in class_names:
            print(f"✅ Found class: {cls}")
        else:
            print(f"❌ Missing class: {cls}")
    
    # Check imports
    required_imports = [
        'argparse',
        'json', 
        'sys',
        'datetime',
        'pathlib'
    ]
    
    import_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                import_names.append(node.module)
    
    for imp in required_imports:
        if any(imp in name for name in import_names):
            print(f"✅ Found import: {imp}")
        else:
            print(f"⚠️  Import may be missing: {imp}")
    
    print(f"\n📊 Summary:")
    print(f"   Functions: {len(function_names)} found")
    print(f"   Classes: {len(class_names)} found")
    print(f"   Imports: {len(import_names)} found")
    
    return len(missing_functions) == 0

if __name__ == "__main__":
    success = test_cli_structure()
    sys.exit(0 if success else 1)