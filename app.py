"""Flask web application for Titanic survival prediction.

This application provides a web interface for predicting passenger survival
using a trained machine learning model. Users input passenger characteristics
and receive a survival probability prediction.
"""

from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MODEL_PATH = "model/titanic_survival_model.pkl"
PORT = int(os.environ.get("PORT", 5000))
DEBUG_MODE = os.environ.get("FLASK_ENV", "development") == "development"

# Load trained model
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}")
    model = None


def format_passenger_input(form_data):
    """Convert form data to structured passenger record.
    
    Args:
        form_data: Flask request form object
        
    Returns:
        dict: Passenger characteristics
        
    Raises:
        ValueError: If form data is invalid
    """
    try:
        return {
            "Pclass": int(form_data["Pclass"]),
            "Sex": form_data["Sex"],
            "Age": float(form_data["Age"]),
            "Fare": float(form_data["Fare"]),
            "Embarked": form_data["Embarked"]
        }
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid form data: {str(e)}")


def make_prediction(passenger_data):
    """Generate survival prediction for passenger.
    
    Args:
        passenger_data: dict with passenger characteristics
        
    Returns:
        str: Human-readable prediction result
    """
    if model is None:
        return "Error: Model not loaded"
    
    try:
        sample_df = pd.DataFrame([passenger_data])
        prediction = model.predict(sample_df)[0]
        return "Survived ✅" if prediction == 1 else "Did Not Survive ❌"
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error: Could not generate prediction"


@app.route("/", methods=["GET", "POST"])
def home():
    """Handle home page GET/POST requests.
    
    Returns:
        Rendered template with prediction result (if available)
    """
    result = None
    error = None

    if request.method == "POST":
        try:
            passenger_data = format_passenger_input(request.form)
            result = make_prediction(passenger_data)
        except ValueError as e:
            error = str(e)
            logger.warning(f"Form validation error: {error}")

    return render_template("index.html", result=result, error=error)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors gracefully."""
    return render_template("index.html", error="Page not found"), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors gracefully."""
    logger.error(f"Server error: {str(error)}")
    return render_template("index.html", error="Server error occurred"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
