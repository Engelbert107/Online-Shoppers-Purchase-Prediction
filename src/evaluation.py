from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall_pos = recall_score(y, y_pred)
    recall_neg = recall_score(y, y_pred, pos_label=0)

    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Recall+ : {recall_pos:.4f}, Recall- : {recall_neg:.4f}")

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
