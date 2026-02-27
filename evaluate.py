from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data_loader import load_data, split_features_target, split_data
from src.reporting import save_classification_report, save_confusion_matrix_plot, save_roc_curve
from src.model_registry import load_latest_model
from src.logger import get_logger

logger = get_logger("EVALUATION")


def main():
    logger.info("========== EVALUATION STARTED ==========")

    # Load model
    pipeline = load_latest_model()

    # Load data
    df = load_data()
    X, y = split_features_target(df)  # y stays boolean: True/False
    _, X_test, _, y_test = split_data(X, y)
    logger.info(f"Test set prepared: {X_test.shape}")

    # Predict
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ------------------------
    # Compute metrics
    # ------------------------
    accuracy = accuracy_score(y_test, y_pred) * 100
    tpr = recall_score(y_test, y_pred, pos_label=True)
    tnr = recall_score(y_test, y_pred, pos_label=False)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Display metrics
    print("\n================ METRICS ================\n")
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print(f"Accuracy           : {accuracy:.2f}%")
    print(f"F1 Score           : {f1:.4f}")
    print(f"True Positive Rate : {tpr:.4f}")
    print(f"True Negative Rate : {tnr:.4f}")
    

    # ------------------------
    # Classification report with original labels
    # ------------------------
    print("\nClassification Report:\n")
    print(classification_report(
        y_test, 
        y_pred, 
        labels=[False, True],
        target_names=["False", "True"]
    ))

    # ------------------------
    # Confusion matrix with original labels
    # ------------------------
    cm = confusion_matrix(y_test, y_pred, labels=[False, True])
    print("\nConfusion Matrix:\n")
    print(cm)

    # ------------------------
    # Save artifacts 
    # ------------------------
    save_classification_report(y_test, y_pred, "latest")
    save_confusion_matrix_plot(y_test, y_pred)
    save_roc_curve(y_test, y_prob)

    logger.info("========== EVALUATION FINISHED ==========")


if __name__ == "__main__":
    main()
