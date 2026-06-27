import logging
from sklearn.metrics import accuracy_score, classification_report
from src.config import appconfig

# FIXED: Read 'accuracy' threshold instead of 'r2' from config.ini
accuracy_min = float(appconfig['Evaluation']['accuracy'])


def get_eval_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for classification.
    """
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(accuracy)
    }


def run(y_true, y_pred):
    """
    Determine if the model meets the minimum accuracy threshold for deployment.
    """
    metrics = get_eval_metrics(y_true, y_pred)
    current_accuracy = metrics["accuracy"]

    logging.info(f"Current Model Accuracy: {current_accuracy:.4f} (Required Minimum: {accuracy_min:.4f})")

    # Detailed classification report printed out in logs
    logging.info("\n" + classification_report(y_true, y_pred))

    if current_accuracy >= accuracy_min:
        logging.info("Model performance passed the minimum threshold.")
        return True
    else:
        logging.warning("Model performance failed to meet the minimum threshold.")
        return False
