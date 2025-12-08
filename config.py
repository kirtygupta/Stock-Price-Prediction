import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-this-in-production-12345'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'stock_predictor.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = 1800  # 30 minutes
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    
    # Stock prediction settings
    STOCK_DATA_DIR = os.path.join(basedir, 'data')
    MODEL_DIR = os.path.join(basedir, 'stock_models')
    
    # ML model settings
    PREDICTION_DAYS_DEFAULT = 30
    TRAIN_TEST_SPLIT = 0.8
    
    # API settings
    YAHOO_FINANCE_TIMEOUT = 30
    MAX_PREDICTION_DAYS = 365
    
    # Debug and testing
    DEBUG = False
    TESTING = False
    
    # Logging
    LOG_FILE = os.path.join(basedir, 'stock_predictor.log')
    LOG_LEVEL = 'INFO'
    
    # Cache
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    def init_app(self):
        """Initialize application with configuration"""
        # Create necessary directories
        for directory in [self.UPLOAD_FOLDER, self.STOCK_DATA_DIR, self.MODEL_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Development database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'stock_predictor_dev.db')
    
    # Development-specific settings
    EXPLAIN_TEMPLATE_LOADING = False
    TEMPLATES_AUTO_RELOAD = True
    
    # Disable caching in development
    CACHE_TYPE = 'null'
    
    # Allow insecure connections in development
    SESSION_COOKIE_SECURE = False

class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    
    # Test database (in-memory)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF protection in testing
    WTF_CSRF_ENABLED = False
    
    # Test-specific settings
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    
    # Use a fixed secret key for testing
    SECRET_KEY = 'test-secret-key-12345'
    
    # Disable logging during tests
    LOG_LEVEL = 'CRITICAL'

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    
    # Production database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'stock_predictor_prod.db')
    
    # Security settings for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    # Cache settings for production
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 600
    
    # Disable template auto-reload in production
    TEMPLATES_AUTO_RELOAD = False
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Create file handler for production logging
        file_handler = RotatingFileHandler(
            cls.LOG_FILE,
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        file_handler.setLevel(getattr(logging, cls.LOG_LEVEL))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to app logger
        app.logger.addHandler(file_handler)
        app.logger.setLevel(getattr(logging, cls.LOG_LEVEL))
        
        # Log startup
        app.logger.info('Stock Predictor startup')

class DockerConfig(ProductionConfig):
    """Configuration for Docker deployment"""
    @classmethod
    def init_app(cls, app):
        ProductionConfig.init_app(app)
        
        # Docker-specific logging
        import logging
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        app.logger.addHandler(console)

class HerokuConfig(ProductionConfig):
    """Configuration for Heroku deployment"""
    
    @classmethod
    def init_app(cls, app):
        ProductionConfig.init_app(app)
        
        # Heroku-specific settings
        import logging
        from logging import StreamHandler
        
        # Log to stdout for Heroku
        stream_handler = StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)
        
        # Handle proxy headers
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Create configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'docker': DockerConfig,
    'heroku': HerokuConfig,
    'default': DevelopmentConfig
}

# Helper function to get configuration
def get_config(config_name=None):
    """Get configuration class by name"""
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'default')
    
    return config.get(config_name, DevelopmentConfig)

# Environment variable documentation
ENV_VARS = {
    'FLASK_CONFIG': 'Configuration to use (development, testing, production)',
    'SECRET_KEY': 'Secret key for session security',
    'DATABASE_URL': 'Database connection URL',
    'DEV_DATABASE_URL': 'Development database URL',
    'LOG_LEVEL': 'Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
}

def print_config_summary(config_obj):
    """Print a summary of the current configuration"""
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    
    # Hide sensitive values
    hidden_keys = ['SECRET_KEY', 'DATABASE_URL', 'DEV_DATABASE_URL']
    
    for key in dir(config_obj):
        if not key.startswith('_') and key.isupper():
            value = getattr(config_obj, key)
            
            # Hide sensitive information
            if key in hidden_keys and value:
                if 'sqlite' in str(value):
                    value = 'sqlite:///...[database path]'
                elif key == 'SECRET_KEY':
                    value = '***HIDDEN***' if len(str(value)) > 10 else value
                else:
                    value = '***HIDDEN***'
            
            print(f"{key:30} = {value}")
    
    print("=" * 60)

# Example .env file content
ENV_EXAMPLE = """# Stock Predictor Configuration
# Copy this to .env file and modify as needed

# Application Configuration
FLASK_CONFIG=development
SECRET_KEY=your-secret-key-here-change-in-production

# Database Configuration
DATABASE_URL=sqlite:///stock_predictor.db
DEV_DATABASE_URL=sqlite:///stock_predictor_dev.db

# Logging
LOG_LEVEL=INFO

# For production with PostgreSQL:
# DATABASE_URL=postgresql://username:password@localhost/stock_predictor

# For production with MySQL:
# DATABASE_URL=mysql://username:password@localhost/stock_predictor
"""

def create_env_file():
    """Create a sample .env file if it doesn't exist"""
    env_path = os.path.join(basedir, '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write(ENV_EXAMPLE)
        print(f"✅ Created sample .env file at {env_path}")
        print("⚠️  Please edit the .env file with your configuration")
    else:
        print(f"📁 .env file already exists at {env_path}")
    
    return env_path

if __name__ == '__main__':
    # Create .env file if it doesn't exist
    create_env_file()
    
    # Load and print default configuration
    cfg = get_config('default')
    print_config_summary(cfg())