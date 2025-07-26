import pandas as pd
import numpy as np

def test_csv_processing():
    """Test the CSV processing to ensure no whitespace issues"""
    try:
        # Read the sample CSV file
        df = pd.read_csv('sample_customers_for_prediction.csv')
        print(f"✅ Successfully read CSV file with {len(df)} records")
        
        # Check for any trailing spaces in string columns
        string_columns = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
        
        print("\n🔍 Checking for whitespace issues...")
        for col in string_columns:
            if col in df.columns:
                # Check for trailing spaces
                has_trailing_spaces = df[col].astype(str).str.endswith(' ').any()
                has_leading_spaces = df[col].astype(str).str.startswith(' ').any()
                
                if has_trailing_spaces or has_leading_spaces:
                    print(f"⚠️  Found whitespace issues in column: {col}")
                else:
                    print(f"✅ Column {col}: Clean")
        
        # Test data cleaning
        print("\n🧹 Testing data cleaning...")
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Check unique values in poutcome column
        unique_poutcomes = df['poutcome'].unique()
        print(f"\n📊 Unique poutcome values: {unique_poutcomes}")
        
        # Check for any remaining whitespace issues
        print("\n🔍 Final check for whitespace...")
        for col in string_columns:
            if col in df.columns:
                has_issues = df[col].astype(str).str.contains(r'^\s|\s$').any()
                if has_issues:
                    print(f"❌ Still has whitespace issues: {col}")
                else:
                    print(f"✅ {col}: Clean after processing")
        
        print(f"\n🎉 CSV processing test completed successfully!")
        print(f"📈 Sample data summary:")
        print(f"   - Total records: {len(df)}")
        print(f"   - Age range: {df['age'].min()} - {df['age'].max()}")
        print(f"   - Balance range: ${df['balance'].min():,.0f} - ${df['balance'].max():,.0f}")
        print(f"   - Job categories: {df['job'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing CSV: {str(e)}")
        return False

if __name__ == "__main__":
    test_csv_processing() 