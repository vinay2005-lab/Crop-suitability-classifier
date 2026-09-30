from pathlib import Path
import sys
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.download_dataset import download_dataset
from src.preprocessing import FEATURES, TARGET, load_and_clean_data

DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for directory in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def make_preprocessor(scale=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)

def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(scale=True)),
            ("classifier", LogisticRegression(
                max_iter=3000, random_state=RANDOM_STATE
            )),
        ]),
        "KNN": Pipeline([
            ("preprocessor", make_preprocessor(scale=True)),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", make_preprocessor(scale=False)),
            ("classifier", DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE
            )),
        ]),
    }

def create_eda(df):
    plt.figure(figsize=(12, 7))
    counts = df[TARGET].value_counts().sort_values()
    plt.barh(counts.index, counts.values)
    plt.title("Crop Class Distribution")
    plt.xlabel("Number of Samples")
    plt.ylabel("Crop")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        axes[i].hist(df[feature].dropna(), bins=25, edgecolor="black")
        axes[i].set_title(f"Distribution: {feature}")
    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=180)
    plt.close(fig)

    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        axes[i].boxplot(df[feature].dropna(), vert=True)
        axes[i].set_title(f"Boxplot: {feature}")
    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Outlier Inspection", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "boxplots.png", dpi=180)
    plt.close(fig)

def write_report(df, duplicate_count, comparison_df, final_name, best_metrics):
    report = (
        "# CropSuit Technical Report — Generated Training Summary\n\n"
        "## Abstract\n\n"
        "This project implements a supervised multi-class machine-learning "
        "prototype for crop suitability classification. The system uses soil "
        "nutrient and environmental measurements to predict a crop category "
        "and exposes the trained model through a Streamlit application.\n\n"
        "## Dataset\n\n"
        f"- Rows after duplicate removal: {len(df)}\n"
        f"- Crop classes: {df[TARGET].nunique()}\n"
        f"- Input features: {', '.join(FEATURES)}\n"
        f"- Duplicate rows removed: {duplicate_count}\n\n"
        "## Data Quality\n\n"
        "The dataset was checked for required columns, numeric conversion, "
        "missing values, duplicates and class distribution. Missing numerical "
        "values, if present, are handled by median imputation inside the model pipeline.\n\n"
        "## Exploratory Data Analysis\n\n"
        "The project generates class distribution, feature distributions, "
        "a correlation heatmap and boxplots for outlier inspection.\n\n"
        "## Methodology\n\n"
        "The data is split into training and testing subsets using a stratified "
        "80/20 split. Three foundational classifiers are compared: Logistic "
        "Regression, K-Nearest Neighbors and Decision Tree. Preprocessing is "
        "included inside scikit-learn pipelines to reduce the risk of data leakage.\n\n"
        "## Model Comparison\n\n"
        "```text\n"
        + comparison_df.to_string(index=False)
        + "\n```\n\n"
        f"## Selected Model\n\n**{final_name}**\n\n"
        "The model was selected using the highest mean 5-fold cross-validation macro F1 on the training data. The held-out test set is used only for final evaluation.\n\n"
        "## Final Test Metrics\n\n"
        f"- Accuracy: {best_metrics['accuracy']:.4f}\n"
        f"- Macro Precision: {best_metrics['precision']:.4f}\n"
        f"- Macro Recall: {best_metrics['recall']:.4f}\n"
        f"- Macro F1: {best_metrics['f1']:.4f}\n\n"
        "## Discussion\n\n"
        "Accuracy describes overall correctness. Macro precision, recall and "
        "F1 give equal weight to every crop class and help inspect performance "
        "across the multi-class target.\n\n"
        "## Limitations\n\n"
        "The dataset is a public educational dataset and should not be interpreted "
        "as a complete real-world agronomic decision system.\n\n"
        "## Conclusion\n\n"
        "The completed pipeline demonstrates data preparation, EDA, preprocessing, "
        "model comparison, evaluation, model persistence and interactive prediction."
    )
    (REPORTS_DIR / "technical_paper_generated.md").write_text(
        report, encoding="utf-8"
    )

def main():
    print("=" * 70)
    print("CropSuit - Crop Suitability Classification")
    print("=" * 70)

    if not DATA_PATH.exists():
        download_dataset()

    df, duplicate_count = load_and_clean_data(DATA_PATH)

    print(f"Dataset shape after duplicate removal: {df.shape}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print("Missing values before pipeline imputation:")
    print(df.isna().sum())
    print(f"Number of crop classes: {df[TARGET].nunique()}")

    create_eda(df)

    X = df[FEATURES]
    y_text = df[TARGET]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = make_models()
    results = []
    trained = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        # Model selection is performed only on the training portion using
        # 5-fold macro-F1 cross-validation. The held-out test set remains
        # untouched until final evaluation.
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "model": name,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "f1": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
        }
        results.append(metrics)
        trained[name] = (model, y_pred)

        print(
            f"CV Macro F1={metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f} | "
            f"Test Accuracy={metrics['accuracy']:.4f} | "
            f"Test Precision={metrics['precision']:.4f} | "
            f"Test Recall={metrics['recall']:.4f} | "
            f"Test F1={metrics['f1']:.4f}"
        )

    comparison_df = pd.DataFrame(results).sort_values(
        by="cv_f1_mean", ascending=False
    ).reset_index(drop=True)

    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    final_name = comparison_df.iloc[0]["model"]
    final_model, final_pred = trained[final_name]
    final_metrics = comparison_df.iloc[0].to_dict()

    report_text = classification_report(
        y_test,
        final_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    (RESULTS_DIR / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    cm = confusion_matrix(y_test, final_pred)
    plt.figure(figsize=(12, 10))
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_encoder.classes_)))
    ax.set_yticks(range(len(label_encoder.classes_)))
    ax.set_xticklabels(label_encoder.classes_, rotation=45, ha="right")
    ax.set_yticklabels(label_encoder.classes_)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(f"Confusion Matrix — {final_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    artifact = {
        "model": final_model,
        "label_encoder": label_encoder,
        "features": FEATURES,
        "model_name": final_name,
        "metrics": {
            "accuracy": float(final_metrics["accuracy"]),
            "precision": float(final_metrics["precision"]),
            "recall": float(final_metrics["recall"]),
            "f1": float(final_metrics["f1"]),
        },
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        artifact,
        MODEL_DIR / "crop_suitability_model.joblib"
    )

    summary = {
        "selected_model": final_name,
        "metrics": artifact["metrics"],
        "dataset_rows": int(len(df)),
        "dataset_columns": list(df.columns),
        "classes": label_encoder.classes_.tolist(),
        "duplicates_removed": int(duplicate_count),
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    metrics_text = (
        f"Selected model: {final_name}\n"
        f"Accuracy: {final_metrics['accuracy']:.4f}\n"
        f"Macro Precision: {final_metrics['precision']:.4f}\n"
        f"Macro Recall: {final_metrics['recall']:.4f}\n"
        f"Macro F1: {final_metrics['f1']:.4f}\n"
    )
    (RESULTS_DIR / "metrics.txt").write_text(
        metrics_text, encoding="utf-8"
    )

    write_report(
        df,
        duplicate_count,
        comparison_df,
        final_name,
        {
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1": final_metrics["f1"],
        },
    )

    print("\n" + "=" * 70)
    print(f"FINAL MODEL: {final_name}")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1:        {final_metrics['f1']:.4f}")
    print("=" * 70)
    print("Model saved to:", MODEL_DIR / "crop_suitability_model.joblib")
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
