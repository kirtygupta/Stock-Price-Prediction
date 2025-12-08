import os
import shutil

print("Fixing CSV file names for dashboard...")
print("="*60)

data_dir = 'data'

# First rename MM to M_M
if os.path.exists(os.path.join(data_dir, 'MM_historical.csv')):
    shutil.move(
        os.path.join(data_dir, 'MM_historical.csv'),
        os.path.join(data_dir, 'M_M_historical.csv')
    )
    print("✅ Renamed MM_historical.csv to M_M_historical.csv")

# Create copies with display names
display_copies = {
    'RELIANCE_historical.csv': 'RELIANCE_historical.csv',  # Already correct
    'TCS_historical.csv': 'TCS_historical.csv',
    'HDFCBANK_historical.csv': 'HDFC_historical.csv',
    'ICICIBANK_historical.csv': 'ICICI_historical.csv',
    'BHARTIARTL_historical.csv': 'BHARTI_AIRTEL_historical.csv',
    'SBIN_historical.csv': 'SBI_historical.csv',
    'INFY_historical.csv': 'INFOSYS_historical.csv',
    'LICI_historical.csv': 'LIC_historical.csv',
    'HINDUNILVR_historical.csv': 'HINDUNILVER_historical.csv',
    'ITC_historical.csv': 'ITC_historical.csv',
    'LT_historical.csv': 'LT_historical.csv',
    'HCLTECH_historical.csv': 'HCLTECH_historical.csv',
    'BAJFINANCE_historical.csv': 'BAJFINANCE_historical.csv',
    'SUNPHARMA_historical.csv': 'SUNPHARMA_historical.csv',
    'M_M_historical.csv': 'M_M_historical.csv',
    'MARUTI_historical.csv': 'MARUTI_historical.csv',
    'KOTAKBANK_historical.csv': 'KOTAK_historical.csv',
    'AXISBANK_historical.csv': 'AXISBANK_historical.csv',
    'ULTRACEMCO_historical.csv': 'ULTRACEMCO_historical.csv',
    'TATAMOTORS_historical.csv': 'TATA_MOTORS_historical.csv',
    'ONGC_historical.csv': 'ONGC_historical.csv',
    'NTPC_historical.csv': 'NTPC_historical.csv',
    'TITAN_historical.csv': 'TITAN_historical.csv',
    'ADANIENT_historical.csv': 'ADANI_historical.csv',
    'COALINDIA_historical.csv': 'COALINDIA_historical.csv'
}

created_count = 0
for actual_file, display_file in display_copies.items():
    actual_path = os.path.join(data_dir, actual_file)
    display_path = os.path.join(data_dir, display_file)
    
    if os.path.exists(actual_path):
        if not os.path.exists(display_path) or actual_file != display_file:
            shutil.copy2(actual_path, display_path)
            created_count += 1
            if actual_file != display_file:
                print(f"✅ Created: {display_file} (from {actual_file})")

print(f"\n📊 Created {created_count} display files")
print("\n" + "="*60)
print("✅ CSV files fixed successfully!")
print("Now run: python app.py")
print("="*60)