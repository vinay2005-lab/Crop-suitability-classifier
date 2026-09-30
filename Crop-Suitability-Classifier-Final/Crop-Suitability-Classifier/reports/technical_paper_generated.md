# CropSuit Technical Report — Generated Training Summary

## Abstract

This project implements a supervised multi-class machine-learning prototype for crop suitability classification. The system uses soil nutrient and environmental measurements to predict a crop category and exposes the trained model through a Streamlit application.

## Dataset

- Rows after duplicate removal: 2200
- Crop classes: 22
- Input features: N, P, K, temperature, humidity, ph, rainfall
- Duplicate rows removed: 0

## Data Quality

The dataset was checked for required columns, numeric conversion, missing values, duplicates and class distribution. Missing numerical values, if present, are handled by median imputation inside the model pipeline.

## Exploratory Data Analysis

The project generates class distribution, feature distributions, a correlation heatmap and boxplots for outlier inspection.

## Methodology

The data is split into training and testing subsets using a stratified 80/20 split. Three foundational classifiers are compared: Logistic Regression, K-Nearest Neighbors and Decision Tree. Preprocessing is included inside scikit-learn pipelines to reduce the risk of data leakage.

## Model Comparison

```text
              model  cv_f1_mean  cv_f1_std  accuracy  precision   recall       f1
      Decision Tree    0.982782   0.006087  0.979545   0.981011 0.979545 0.979412
Logistic Regression    0.967442   0.011470  0.972727   0.974022 0.972727 0.972464
                KNN    0.965991   0.008004  0.979545   0.980356 0.979545 0.979283
```

## Selected Model

**Decision Tree**

The model was selected using the highest mean 5-fold cross-validation macro F1 on the training data. The held-out test set is used only for final evaluation.

## Final Test Metrics

- Accuracy: 0.9795
- Macro Precision: 0.9810
- Macro Recall: 0.9795
- Macro F1: 0.9794

## Discussion

Accuracy describes overall correctness. Macro precision, recall and F1 give equal weight to every crop class and help inspect performance across the multi-class target.

## Limitations

The dataset is a public educational dataset and should not be interpreted as a complete real-world agronomic decision system.

## Conclusion

The completed pipeline demonstrates data preparation, EDA, preprocessing, model comparison, evaluation, model persistence and interactive prediction.