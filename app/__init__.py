import os
from flask import Flask
from flask_pymongo import PyMongo
from dotenv import load_dotenv

# Find the .env file in the root directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Initialize database extension
mongo = PyMongo()

def create_app():
    """Application factory function"""
    app = Flask(__name__)
    
    # Configure the MongoDB Atlas connection string
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not set in .env file. Please add your MongoDB Atlas connection string.")
    
    app.config["MONGO_URI"] = mongo_uri

    # Initialize extensions with the app
    mongo.init_app(app)

    # Register the main routes/endpoints
    from app import routes
    app.register_blueprint(routes.bp)

    print("Flask application created successfully with MongoDB.")
    
    # --- NEW: Test the database connection on startup ---
    try:
        # mongo.cx is the client. .server_info() pings the server.
        mongo.cx.server_info()
        print("MongoDB connection successful.")
    except Exception as e:
        print(f"!!! MONGODB CONNECTION FAILED !!!")
        print(f"Error: {e}")
        print("Please check two things:")
        print("1. Is your password correct in the .env file (and percent-encoded)?")
        print("2. Did you add your computer's IP address to the 'Network Access' list in MongoDB Atlas?")
        # Re-raise the error to stop the app from running
        raise e
    # --- End of new code ---

    return app