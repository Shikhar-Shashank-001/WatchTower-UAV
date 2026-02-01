import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# LOAD DATA
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# METRIC FUNCTIONS
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return accuracy, sensitivity, specificity, cm


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Tank", "Tank"],
        yticklabels=["Non-Tank", "Tank"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# CROSS-VALIDATION SETUP
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def cross_val_with_progress(model, X, y, cv, desc):
    y_pred = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(cv.split(X, y), total=cv.get_n_splits(), desc=desc)
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)

    return y_pred

# SVM with cubic kernel
svm_poly = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="poly", degree=3, C=1.0))
])

y_pred_poly = cross_val_with_progress(
    svm_poly, X, y, cv, desc="Cubic SVM Cross-Validation"
)

acc_p, sen_p, spe_p, cm_p = compute_metrics(y, y_pred_poly)

print("\n=== Cubic SVM (Polynomial) ===")
print(f"Accuracy    : {acc_p:.4f}")
print(f"Recall : {sen_p:.4f}")
print(f"Specificity : {spe_p:.4f}")

plot_confusion_matrix(cm_p, "Confusion Matrix – Cubic SVM")

# Train final model & save
svm_poly.fit(X, y)
joblib.dump(svm_poly, "svm_cubic_model.joblib")
print("Saved: svm_cubic_model.joblib")



# RBF (Gaussian) SVM
svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

y_pred_rbf = cross_val_with_progress(
    svm_rbf, X, y, cv, desc="RBF SVM Cross-Validation"
)

acc_r, sen_r, spe_r, cm_r = compute_metrics(y, y_pred_rbf)

print("\n=== RBF (Gaussian) SVM ===")
print(f"Accuracy    : {acc_r:.4f}")
print(f"Recall : {sen_r:.4f}")
print(f"Specificity : {spe_r:.4f}")

plot_confusion_matrix(cm_r, "Confusion Matrix – RBF SVM")

# Train final model & save
svm_rbf.fit(X, y)
joblib.dump(svm_rbf, "svm_rbf_model.joblib")
print("Saved: svm_rbf_model.joblib")
