# CropSuit: Crop Suitability Classification Using Supervised Machine Learning

## Abstract

CropSuit is an educational machine-learning system for classifying crop categories from soil and environmental observations. The system uses nitrogen, phosphorus, potassium, temperature, humidity, soil pH and rainfall as input features. The workflow includes data quality inspection, exploratory data analysis, preprocessing, model comparison, evaluation and an interactive Streamlit prototype.

## 1. Introduction

Agricultural decisions depend on soil and environmental conditions. Data-driven classification can be used as a decision-support prototype to study relationships between measurable conditions and crop categories.

## 2. Problem Definition

The objective is to classify an observation into a suitable crop category using measurable soil and environmental indicators.

## 3. Dataset

The project uses the public Crop Recommendation Dataset. It contains 2,200 observations, seven numerical input features and 22 crop classes.

Dataset source:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

## 4. Methodology

### 4.1 Data Preparation

The dataset is loaded with Pandas. Column names are normalized, required fields are validated, numeric features are converted to numeric values, and exact duplicate rows are removed.

### 4.2 Exploratory Data Analysis

The project examines class distribution, feature distributions, correlations, potential outliers and data completeness.

### 4.3 Preprocessing

The dataset is divided into training and testing sets using a stratified 80/20 split. Median imputation and standardization are placed inside scikit-learn pipelines where required, preventing test-set information from being used during training.

### 4.4 Models

Three foundational classifiers are compared:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree

### 4.5 Evaluation

The following metrics are reported:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion Matrix

The final model is selected using macro F1 on the held-out test set.

## 5. Results

Run:

```bash
python src/train_model.py
```

Then copy the actual generated metrics from:

`results/model_comparison.csv`

and:

`results/metrics.txt`

The script also generates `reports/technical_paper_generated.md` containing the actual results.

## 6. Streamlit Prototype

The Streamlit interface accepts the seven model features and displays the predicted crop and model probability when available. The same saved preprocessing/model pipeline used during training is used for inference.

## 7. Discussion

Accuracy describes overall correctness, while macro precision, recall and F1 provide equal weighting to the individual crop classes.

## 8. Limitations

The system is an educational prototype. The dataset does not represent every soil type, climate, crop variety or local farming condition. Real crop decisions should involve local measurements and agricultural expertise.

## 9. Future Scope

Possible future extensions include regional weather data, soil sensor integration, crop yield prediction, fertilizer recommendation, explainable feature-level recommendations, mobile deployment and larger region-specific datasets.

## 10. Conclusion

CropSuit demonstrates an end-to-end supervised learning workflow for crop suitability classification, from data preparation and EDA through model evaluation and interactive deployment.

## References

1. Crop Recommendation Dataset, Kaggle:
   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Public CSV mirror:
   https://raw.githubusercontent.com/the-amazing-atharva/Crop-Recommendation/main/Crop_recommendation.csv
3. Scikit-learn:
   https://scikit-learn.org/
4. Streamlit:
   https://docs.streamlit.io/
