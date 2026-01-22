"""Model development and training for Titanic survival prediction.

This script handles data loading, preprocessing, model training, and
evaluation for the Titanic survival prediction system.
"""

import pandas as pd
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "train.csv"
MODEL_OUTPUT_PATH = "titanic_survival_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITERATIONS = 1000

# Feature definitions
INPUT_FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
TARGET_COLUMN = "Survived"
NUMERIC_FEATURES = ["Pclass", "Age", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def load_and_prepare_data(data_path):
    """Load Titanic dataset and select relevant features.
    
    Args:
        data_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset with selected features
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Select features and target
    df = df[INPUT_FEATURES + [TARGET_COLUMN]]
    logger.info(f"Dataset shape: {df.shape}")
    
    return df


def handle_missing_values(df):
    """Impute missing values in dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    logger.info("Handling missing values")
    
    # Fill Age with median
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    logger.info(f"Filled Age with median: {age_median}")
    
    # Fill Embarked with mode (most common)
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    logger.info(f"Filled Embarked with mode: {embarked_mode}")
    
    return df


def build_preprocessing_pipeline():
    """Create feature preprocessing pipeline.
    
    Returns:
        ColumnTransformer: Configured preprocessor for scaling/encoding
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def build_model_pipeline(preprocessor):
    """Create complete ML pipeline (preprocessing + model).
    
    Args:
        preprocessor: ColumnTransformer for feature processing
        
    Returns:
        Pipeline: Complete ML pipeline
    """
    model = LogisticRegression(max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE)
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    return pipeline


def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """Train model and generate evaluation metrics.
    
    Args:
        pipeline: Sklearn Pipeline
        X_train, X_test: Training and test feature sets
        y_train, y_test: Training and test target sets
    """
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline


def save_and_verify_model(pipeline, output_path):
    """Save model and verify it can be reloaded.
    
    Args:
        pipeline: Trained sklearn Pipeline
        output_path (str): Where to save the model
    """
    logger.info(f"Saving model to {output_path}")
    joblib.dump(pipeline, output_path)
    
    # Verify model can be loaded
    loaded_pipeline = joblib.load(output_path)
    logger.info(f"Model successfully saved and verified")
    
    return loaded_pipeline


def main():
    """Execute full model development pipeline."""
    try:
        # Load and prepare data
        df = load_and_prepare_data(DATA_PATH)
        df = handle_missing_values(df)
        
        # Prepare features and target
        X = df[INPUT_FEATURES]
        y = df[TARGET_COLUMN]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
        
        # Build and train model
        preprocessor = build_preprocessing_pipeline()
        pipeline = build_model_pipeline(preprocessor)
        pipeline = train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        
        # Save model
        pipeline = save_and_verify_model(pipeline, MODEL_OUTPUT_PATH)
        
        logger.info("Model development complete!")
        
    except Exception as e:
        logger.error(f"Error during model development: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
