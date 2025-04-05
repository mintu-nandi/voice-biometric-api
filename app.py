import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# SQLAlchemy setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.config.from_object('config')
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = app.config['DATABASE_URI']
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the database with the app
db.init_app(app)

# Setup routes and models after app is created
with app.app_context():
    # Import models and routes here to avoid circular imports
    import models
    import routes
    
    # Create all tables
    db.create_all()

    # Register blueprints
    from routes import api_bp
    app.register_blueprint(api_bp)
    
    # Root route to redirect to API documentation
    @app.route('/')
    def root():
        from flask import redirect
        return redirect('/api/')
