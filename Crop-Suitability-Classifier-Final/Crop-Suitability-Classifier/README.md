# CropSuit — Crop Suitability Classification

**Internship Track 1 Capstone — Problem 19**

> Classify whether environmental and soil conditions are suitable for selected crop categories.

## 1. Project Overview

CropSuit is an educational machine-learning prototype that predicts a suitable crop category from seven measurable soil/environmental inputs:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The project follows an end-to-end workflow:

**Dataset → Data Quality → EDA → Preprocessing → Model Comparison → Evaluation → Saved Model → Streamlit Application**

The implementation deliberately uses foundational supervised-learning methods:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

The final model is selected using 5-fold cross-validation macro F1 on the training data. The held-out test set is reserved for final evaluation.

## 2. Dataset

The project uses the public **Crop Recommendation Dataset**, containing 2,200 observations, seven numerical input features, and 22 crop classes.

Dataset fields:

`N, P, K, temperature, humidity, ph, rainfall, label`

The repository includes an automatic downloader because the dataset is an external public resource.

Original/public dataset reference:

- Kaggle Crop Recommendation Dataset: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Public CSV mirror used by the downloader:

- https://raw.githubusercontent.com/the-amazing-atharva/Crop-Recommendation/main/Crop_recommendation.csv

## 3. Project Structure

```text
Crop-Suitability-Classifier/
│
├── data/
│   └── Crop_recommendation.csv       # downloaded automatically
├── models/
│   └── crop_suitability_model.joblib # generated after training
├── notebooks/
│   └── crop_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── download_dataset.py
│   ├── preprocessing.py
│   └── train_model.py
├── results/
├── reports/
├── figures/
├── app.py
├── requirements.txt
├── run_project.bat
├── technical_paper.md
├── VIVA_GUIDE.md
└── README.md
```

## 4. Installation

Open Command Prompt/PowerShell in this project folder:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Train the Project

```bash
python src/train_model.py
```

The script will:

1. Download the dataset if it is not already present.
2. Load and inspect the CSV.
3. Normalize column names and validate required fields.
4. Remove exact duplicate rows.
5. Handle missing numerical values through median imputation inside ML pipelines.
6. Create EDA figures.
7. Split the data using a stratified 80/20 split.
8. Compare Logistic Regression, KNN and Decision Tree.
9. Calculate accuracy, macro precision, macro recall and macro F1.
10. Create a confusion matrix.
11. Save the selected model and label encoder.
12. Generate a results summary and technical-paper draft.

## 6. Run the Streamlit Application

After training:

```bash
streamlit run app.py
```

The browser application accepts the seven input variables and returns:

- Predicted crop
- Prediction probability when available
- Top three model predictions
- Input summary

## 7. Data Leakage Prevention

The train/test split happens before model fitting. Imputation and scaling are performed inside scikit-learn pipelines so that preprocessing statistics are learned only from the training data.

## 8. Model Comparison

| Model | Main idea |
|---|---|
| Logistic Regression | Learns a linear decision boundary |
| KNN | Predicts from nearby training observations |
| Decision Tree | Learns hierarchical feature-based decision rules |

The final model is selected using the mean 5-fold macro F1 calculated only on the training portion; the held-out test set is used only for final evaluation.

## 9. Limitations

This is an educational prototype, not an agronomic advisory service.

A model trained on a public crop-recommendation dataset should not be treated as a guaranteed farming recommendation. Real agricultural decisions can require local soil testing, region-specific climate data, crop variety information, irrigation availability and expert agronomic validation.

## 10. Viva One-Liner

> “Our system is a supervised multi-class classification pipeline that uses soil nutrient and environmental features to predict the crop category associated with a new observation. We perform data-quality checks and EDA, prevent leakage using train/test splitting and pipelines, compare three foundational classifiers, evaluate them using classification metrics, save the selected model, and expose it through Streamlit.”

## 11. Reproducibility

Run:

```bash
python src/train_model.py
streamlit run app.py
```

The model artifact, metrics, figures and generated report are created from the downloaded dataset rather than hard-coded.
