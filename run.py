#!/usr/bin/env python3
import os
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_CONFIG'] = 'development'

from app import app

if __name__ == '__main__':
    with app.app_context():
        from models import db
        db.create_all()
        print("✅ Database initialized")
    
    print("\n🚀 Starting Stock Prediction Platform...")
    print("📊 25 Indian Stocks with ML Models")
    print("🌐 Server: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)