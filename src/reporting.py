import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from config import REPORT_DIR
from src.logger import get_logger

logger = get_logger("REPORTING")


def save_classification_report(y_true, y_pred, model_path):
    os.makedirs(REPORT_DIR, exist_ok=True)

    report = classification_report(y_true, y_pred, output_dict=True)

    report_path = os.path.join(REPORT_DIR, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Classification report saved: {report_path}")


def save_confusion_matrix_plot(y_true, y_pred):
    os.makedirs(REPORT_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()

    path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()

    logger.info(f"Confusion matrix saved: {path}")


def save_roc_curve(y_true, y_prob):
    os.makedirs(REPORT_DIR, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    path = os.path.join(REPORT_DIR, "roc_curve.png")
    plt.savefig(path)
    plt.close()

    logger.info(f"ROC curve saved: {path}")
