import os
import shutil
import sys

def setup_local_environment():
    print("Setting up local environment...")
    print("="*60)
    
    # Create necessary directories
    directories = ['data', 'stock_models', 'stock_models/regression', 'stock_models/classification']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")
    
    # Check for extracted models
    print("\nChecking for model files...")
    
    # Look for .joblib files in stock_models directory
    model_files = []
    for root, dirs, files in os.walk('stock_models'):
        for file in files:
            if file.endswith('.joblib'):
                model_files.append(os.path.join(root, file))
    
    if model_files:
        print(f"✅ Found {len(model_files)} model files")
    else:
        print("⚠️  No model files found. Please extract the stock_models.zip file.")
        print("   The zip file should contain:")
        print("   - stock_models/regression/*.joblib")
        print("   - stock_models/classification/*.joblib")
        print("   - stock_models/*.csv")
        print("   - data/*.csv")
    
    # Check for CSV data files
    csv_files = []
    if os.path.exists('data'):
        for file in os.listdir('data'):
            if file.endswith('.csv'):
                csv_files.append(file)
    
    if csv_files:
        print(f"✅ Found {len(csv_files)} CSV data files")
    else:
        print("⚠️  No CSV data files found in 'data' directory")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the app: python app.py")
    print("3. Access at: http://localhost:5000")
    print("4. Login with: admin / admin123")
    print("="*60)

if __name__ == "__main__":
    setup_local_environment()