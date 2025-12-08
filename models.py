from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        self.last_login = datetime.utcnow()

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stock_symbol = db.Column(db.String(20), nullable=False)
    stock_name = db.Column(db.String(100), nullable=True)
    predicted_price = db.Column(db.Float, nullable=False)
    actual_price = db.Column(db.Float, nullable=True)
    model_used = db.Column(db.String(50), nullable=True)
    prediction_days = db.Column(db.Integer, default=30)
    confidence_score = db.Column(db.Float, nullable=True)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    actual_date = db.Column(db.DateTime, nullable=True)
    
    @property
    def price_difference(self):
        if self.actual_price is None:
            return None
        try:
            return float(self.predicted_price) - float(self.actual_price)
        except (TypeError, ValueError):
            return None
    
    @property
    def difference_percent(self):
        if self.actual_price is None or self.actual_price == 0:
            return None
        try:
            difference = self.price_difference
            if difference is None:
                return None
            return (difference / float(self.actual_price)) * 100
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    
    def to_dict(self):
        return {
            'id': self.id,
            'stock_symbol': self.stock_symbol,
            'stock_name': self.stock_name,
            'predicted_price': self.predicted_price,
            'actual_price': self.actual_price,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'actual_date': self.actual_date.isoformat() if self.actual_date else None,
            'price_difference': self.price_difference,
            'difference_percent': self.difference_percent,
            'model_used': self.model_used,
            'confidence_score': self.confidence_score,
            'prediction_days': self.prediction_days
        }