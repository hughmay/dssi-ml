"""
This module automates model training for the Iris classification dataset.
"""

import argparse
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# CHANGED: Swapped LinearRegression out for a Classifier
from sklearn.linear_model import LogisticRegression 

from src import data_processor
from src import model_registry
from src import evaluation
from src.config import appconfig

logging.basicConfig(level=logging.INFO)

# Extract configurations safely
features = appconfig['Model']['features'].split(',')
# Safely handle if categorical features are empty for Iris
categorical_features_str = appconfig['Model'].get('categorical_features', '')
categorical_features = [f for f in categorical_features_str.split(',') if f]
label = appconfig['Model']['label']

def run(data_path):
    """
    Main script to perform model training.
        Parameters:
            data_path (str): Directory containing the training dataset in csv
        Returns:
            None: No returns required
    """
    logging.info('Process Data...')
    df = data_processor.run(data_path)
    
    # Refined Preprocessor: Iris features are numerical, but we keep 
    # OneHotEncoder infrastructure in case your CSV pipeline injects categorical keys.
    transformers = []
    if categorical_features:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        transformers.append(("cat", categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers, 
        remainder='passthrough'
    )
    
    # Train-Test Split
    logging.info('Start Train-Test Split...')
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], 
        df[label], 
        test_size=appconfig.getfloat('Model', 'test_size'), 
        random_state=0
    )
    
    # Train Classifier
    logging.info('Start Training Classification Model...')
    # CHANGED: Using LogisticRegression with a max_iter ceiling for stability
    clf = LogisticRegression(max_iter=200, random_state=0)
    
    # CHANGED: Updated step name from "regression" to "classification"
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classification", clf)
    ])
    pipeline.fit(X_train, y_train)
    
    # Evaluate and Persist
    logging.info('Evaluating model performance...')
    predictions = pipeline.predict(X_test)
    
    if evaluation.run(y_test, predictions):
        logging.info('Persisting model...')
        mdl_meta = { 
            'name': appconfig['Model']['name'], 
            'metrics': evaluation.get_eval_metrics(y_test, predictions) 
        }
        model_registry.register(pipeline, features, mdl_meta)

    logging.info('Training completed.')
    return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    args = argparser.parse_args()
    run(args.data_path)
