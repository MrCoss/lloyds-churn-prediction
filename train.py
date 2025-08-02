import os
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_FILE = os.path.join("outputs", "prepared_churn_dataset.csv")
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
PLOT_DIR = "plots"
LOG_DIR = "logs"
RANDOM_STATE = 42

# --- Setup environment ---
def setup():
    for folder in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
        os.makedirs(folder, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, "train_phase2.log"), mode='w'),
            logging.StreamHandler()
        ]
    )

# --- Load data ---
def load_data(path):
    logging.info(f"Loading dataset: {path}")
    return pd.read_csv(path)

# --- Generate synthetic validation ---
def create_synthetic_validation(X_train, y_train):
    logging.info("Creating synthetic validation set...")

    def get_class_weights(y):
        pos = sum(y == 1)
        neg = sum(y == 0)
        total = len(y)
        return [neg / total, pos / total]

    weights = get_class_weights(y_train)

    X_syn, y_syn = make_classification(
        n_samples=300,
        n_features=X_train.shape[1],
        n_informative=int(0.7 * X_train.shape[1]),
        n_redundant=0,
        n_clusters_per_class=2,
        weights=weights,
        random_state=42
    )
    return pd.DataFrame(X_syn, columns=X_train.columns), pd.Series(y_syn)

# --- Train models ---
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight='balanced'), {'classifier__C': [0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(class_weight='balanced'), {'classifier__n_estimators': [100, 200]}),
        "Gradient Boosting": (GradientBoostingClassifier(), {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.05, 0.1]}),
        "SVM": (SVC(probability=True, class_weight='balanced'), {'classifier__C': [1, 10], 'classifier__gamma': ['scale']})
    }

    results = {}
    for name, (model, params) in models.items():
        logging.info(f"Tuning: {name}")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", model)
        ])
        clf = GridSearchCV(pipeline, params, cv=5, scoring='roc_auc', n_jobs=-1)
        clf.fit(X_train, y_train)
        results[name] = clf.best_estimator_
        logging.info(f"{name} - Best AUC: {clf.best_score_:.4f}")
    return results

# --- Evaluate models ---
def evaluate(models, X_test, y_test, X_val, y_val):
    metrics = []
    best_auc = -1
    best_model = None
    best_name = ""

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)

        metrics.append({
            "Model": name,
            "Test Accuracy": acc,
            "Test AUC": test_auc,
            "Synthetic Val AUC": val_auc
        })

        # Save report
        with open(os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}_report.txt"), "w") as f:
            f.write(classification_report(y_test, y_pred))

        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model
            best_name = name

    df = pd.DataFrame(metrics).sort_values(by="Test AUC", ascending=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_phase2.csv"), index=False)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_churn_model_phase2.pkl"))
    logging.info(f"Best model: {best_name}, AUC: {best_auc:.4f}")

    # Plots
    plot_roc_curve(best_model, X_test, y_test, best_name)
    plot_confusion_matrix(best_model, X_test, y_test, best_name)
    plot_feature_importance(best_model, X_test.columns, best_name)
    plot_prediction_histogram(best_model, X_test, y_test, best_name)

# --- Plot: ROC Curve ---
def plot_roc_curve(model, X, y, name):
    fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
    auc_score = roc_auc_score(y, model.predict_proba(X)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curve_phase2.png"))
    plt.close()

# --- Plot: Confusion Matrix ---
def plot_confusion_matrix(model, X, y, name):
    cm = confusion_matrix(y, model.predict(X))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix_phase2.png"))
    plt.close()

# --- Plot: Feature Importance ---
def plot_feature_importance(model, feature_names, name):
    classifier = model.named_steps['classifier']
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = 20

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices[:top_n]],
                    y=np.array(feature_names)[indices[:top_n]], palette="viridis")
        plt.title(f"Top {top_n} Feature Importances - {name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "feature_importance_phase2.png"))
        plt.close()

# --- Plot: Prediction Probability Histogram ---
def plot_prediction_histogram(model, X, y, name):
    y_proba = model.predict_proba(X)[:, 1]
    plt.figure(figsize=(8, 5))
    sns.histplot(y_proba[y == 0], label="No Churn", color="blue", kde=True)
    sns.histplot(y_proba[y == 1], label="Churn", color="red", kde=True)
    plt.title(f"Prediction Probability Distribution - {name}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "prediction_histogram_phase2.png"))
    plt.close()

# --- Main ---
def main():
    try:
        setup()
        df = load_data(INPUT_FILE)
        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        X_val, y_val = create_synthetic_validation(X_train, y_train)

        models = train_models(X_train, y_train)
        evaluate(models, X_test, y_test, X_val, y_val)

        logging.info("Phase 2 complete — models trained and evaluated.")
    except Exception as e:
        logging.fatal(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
