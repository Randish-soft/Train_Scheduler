"""
diagnose_csv.py
Diagnostic tool to identify and fix CSV formatting issues
"""
import csv
import pandas as pd
from pathlib import Path

def diagnose_csv(filepath):
    """Diagnose CSV formatting issues"""
    print(f"🔍 Diagnosing CSV: {filepath}\n")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check header
    header = lines[0].strip()
    header_fields = header.split(',')
    num_header_cols = len(header_fields)
    
    print(f"📊 Header Analysis:")
    print(f"   - Number of columns: {num_header_cols}")
    print(f"   - Column names: {header_fields[:5]}... (showing first 5)")
    
    # Find problematic lines
    print(f"\n❌ Problematic lines:")
    problems_found = False
    
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():  # Skip empty lines
            continue
            
        # Try parsing with csv module for proper handling of quoted fields
        try:
            reader = csv.reader([line.strip()])
            row = next(reader)
            num_cols = len(row)
        except:
            num_cols = len(line.strip().split(','))
        
        if num_cols != num_header_cols:
            problems_found = True
            print(f"\n   Line {i}: Has {num_cols} columns (expected {num_header_cols})")
            print(f"   Content preview: {line[:100]}...")
            
            # Show which columns might be extra
            if num_cols > num_header_cols:
                print(f"   ⚠️  Extra columns detected - might have unescaped commas")
    
    if not problems_found:
        print("   ✅ No column count issues found!")
    
    return problems_found

def analyze_csv_content(filepath):
    """Deeper analysis of CSV content"""
    print(f"\n📋 Content Analysis:")
    
    try:
        # Try reading with pandas
        df = pd.read_csv(filepath, nrows=5, on_bad_lines='skip')
        print(f"\n   ✅ Pandas can read the file (with skipping bad lines)")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
    except Exception as e:
        print(f"\n   ❌ Pandas cannot read the file: {e}")
    
    # Check for common issues
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"\n🔍 Common Issues Check:")
    
    # Check for quotes
    if '"' in content:
        print("   - Found double quotes (might have fields with commas)")
    
    # Check for special characters
    special_chars = ['\t', '|', ';']
    for char in special_chars:
        if char in content:
            print(f"   - Found '{char}' character (might be using different delimiter)")

def fix_csv_columns(filepath, output_path=None):
    """Attempt to fix CSV by properly parsing it"""
    if output_path is None:
        output_path = filepath.parent / f"{filepath.stem}_fixed.csv"
    
    print(f"\n🔧 Attempting to fix CSV...")
    
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows.append(header)
        
        for i, row in enumerate(reader, start=2):
            if len(row) == len(header):
                rows.append(row)
            else:
                print(f"   - Skipping line {i} with {len(row)} columns")
    
    # Write fixed CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"   ✅ Fixed CSV saved to: {output_path}")
    print(f"   - Kept {len(rows)-1} valid rows out of original")
    
    return output_path

def main():
    """Main diagnostic function"""
    # Find the CSV file
    csv_path = Path("input/master-data.csv")
    
    if not csv_path.exists():
        print("❌ Error: master-data.csv not found in input/ directory")
        return
    
    print("="*60)
    print("🔧 CSV DIAGNOSTIC TOOL")
    print("="*60)
    
    # Run diagnostics
    has_problems = diagnose_csv(csv_path)
    analyze_csv_content(csv_path)
    
    if has_problems:
        print("\n" + "="*60)
        response = input("\n🔧 Would you like to attempt automatic fix? (y/n): ").strip().lower()
        
        if response == 'y':
            fixed_path = fix_csv_columns(csv_path)
            
            print("\n📝 Next steps:")
            print(f"1. Review the fixed file: {fixed_path}")
            print(f"2. If it looks good, replace the original:")
            print(f"   cp {fixed_path} {csv_path}")
            print(f"3. Run the pipeline again")
    else:
        print("\n✅ CSV appears to be properly formatted!")

if __name__ == "__main__":
    main()