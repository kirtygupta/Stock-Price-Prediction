#!/usr/bin/env python3
"""
Setup script for Stock Prediction Platform
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def create_directories():
    """Create necessary directories"""
    directories = [
        'instance',
        'data',
        'logs',
        'templates',
        'static',
        'static/css',
        'static/js',
        'static/images'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def copy_example_files():
    """Copy example configuration files"""
    example_files = {
        '.env.example': '.env',
        'config.py': 'config.py',
    }
    
    for src, dst in example_files.items():
        if not os.path.exists(dst):
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"✅ Copied: {src} -> {dst}")
            else:
                print(f"⚠️  Example file not found: {src}")

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)

def setup_database():
    """Initialize the database"""
    print("Setting up database...")
    try:
        from app import init_database
        init_database()
        print("✅ Database setup complete")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        sys.exit(1)

def main():
    """Main setup function"""
    print("=" * 50)
    print("Stock Prediction Platform - Setup Script")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Copy example files
    copy_example_files()
    
    # Install requirements
    install_requirements()
    
    # Setup database
    setup_database()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file with your configuration")
    print("2. Make sure you have stock data in the 'data' directory")
    print("3. Make sure you have ML models in the 'stock_models' directory")
    print("4. Run the application: python app.py")
    print("5. Visit http://localhost:5000 in your browser")
    print("=" * 50)

if __name__ == '__main__':
    main()