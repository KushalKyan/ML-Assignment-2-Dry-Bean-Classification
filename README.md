# Dry Bean Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models that can identify the variety of a dry bean based on its morphological characteristics.

The physical characteristics of a bean, such as area, perimeter, axis length, roundness, compactness, and shape factors, can help distinguish one bean variety from another. In this project, these characteristics are used as input features for supervised machine learning models.

The project implements and evaluates the following five classification algorithms:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models are trained and evaluated using the same cleaned dataset and train-test split.

Their performance is compared using the following evaluation metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The trained models are saved and used in a Streamlit web application. The application allows a user to upload test data, select a trained model, generate predictions, and view the performance metrics, confusion matrix, classification report, and prediction results without retraining the models.

---

## b. Dataset Description

### Dataset Name

**Dry Bean Dataset**

### Dataset Source

The dataset was obtained from the UCI Machine Learning Repository:

[UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

### Dataset Summary

The Dry Bean Dataset contains measurements obtained from images of dry bean grains. A computer vision system was used to extract morphological characteristics from each bean image.

The original dataset contains:

| Dataset Property | Value |
|---|---:|
| Number of Instances | 13,611 |
| Number of Input Features | 16 |
| Number of Target Classes | 7 |
| Problem Type | Multiclass Classification |
| Missing Values | 0 |
| Duplicate Rows Identified | 68 |
| Records After Duplicate Removal | 13,543 |
| Training Records | 10,834 |
| Testing Records | 2,709 |
| Train-Test Ratio | 80:20 |

Duplicate rows were removed before model training. No missing values were found in the dataset.

A stratified train-test split was used to maintain a similar class distribution in the training and testing datasets.

The `random_state` was set to `42` to make the experiment reproducible.

### Input Features

The dataset contains the following 16 input features:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

The Excel dataset used for this project contains the column names `AspectRation` and `roundness`. These names were retained to maintain consistency between the dataset, trained models, saved feature names, test file, and Streamlit application.

### Target Variable

The target column is:

`Class`

The target variable contains the following seven dry bean varieties:

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

### Class Distribution After Duplicate Removal

| Bean Class | Number of Records |
|---|---:|
| DERMASON | 3,546 |
| SIRA | 2,636 |
| SEKER | 2,027 |
| HOROZ | 1,860 |
| CALI | 1,630 |
| BARBUNYA | 1,322 |
| BOMBAY | 522 |
| **Total** | **13,543** |

The dataset is moderately imbalanced because the DERMASON class contains substantially more records than the BOMBAY class.

For this reason, weighted Precision, Recall, F1 Score, and AUC were used when comparing model performance.

### Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the Excel dataset using `pandas`.
2. Checked the dataset dimensions and column names.
3. Checked for missing values.
4. Identified and removed 68 duplicate rows.
5. Separated the 16 input features from the target variable.
6. Encoded the textual target classes using `LabelEncoder`.
7. Divided the cleaned dataset into 80% training data and 20% testing data.
8. Used stratification to preserve the class distribution in the training and testing datasets.
9. Fitted `StandardScaler` only on the training data.
10. Applied feature scaling to Logistic Regression, KNN, and SVM.
11. Used the original unscaled features for Decision Tree and Gaussian Naive Bayes.
12. Saved the trained models and preprocessing objects using `joblib`.

The scaler was fitted only on the training data to prevent data leakage.

---

## c. GitHub Repository Link

The complete project source code, dataset, saved models, test data, dependency file, and Streamlit application are available in the following GitHub repository:

[GitHub Repository - ML Assignment 2 Dry Bean Classification](https://github.com/KushalKyan/ML-Assignment-2-Dry-Bean-Classification)

### Live Streamlit Application

The deployed Streamlit application is available at:

[Dry Bean Classification - Streamlit App](https://ml-assignment-2-dry-bean-classification-jusfheef5ty2hyuwdertn3.streamlit.app/)

### Repository Contents

The GitHub repository contains the following files:

```text
ML-Assignment-2-Dry-Bean-Classification/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── Dry_Bean_Dataset.xlsx
├── test_data.csv
├── model_results.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── feature_names.pkl
```

### Main Files

- **`train_models.py`**: Loads and cleans the dataset, trains all five models, calculates the evaluation metrics, saves the trained models, and generates the test data.
- **`app.py`**: Loads the saved models and provides the Streamlit user interface.
- **`test_data.csv`**: Contains the held-out test dataset used to evaluate the trained models.
- **`model_results.csv`**: Contains the evaluation metrics produced by the five models.
- **`requirements.txt`**: Contains the Python dependencies required to run and deploy the project.
- **`model/`**: Contains the five trained models and saved preprocessing objects.
- **`README.md`**: Contains the project documentation, model results, and observations.

---

## d. Models Used

The following five machine learning classification models were implemented.

### 1. Logistic Regression

Logistic Regression is a linear classification algorithm. For this multiclass problem, the model learns the relationship between the scaled morphological features and the seven bean classes.

The maximum number of iterations was increased to allow the optimization process to converge.

### 2. Decision Tree

The Decision Tree classifier creates a tree of decision rules based on the feature values.

A Decision Tree does not require feature scaling because it divides the data according to feature thresholds.

### 3. K-Nearest Neighbors

K-Nearest Neighbors (KNN) classifies a bean according to the classes of its nearest training observations.

The model was configured with **5 neighbors**.

Because KNN is based on distances between observations, standardized input features were used.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm that assumes the input features follow Gaussian distributions within each class.

The model also assumes that the features are conditionally independent given the target class.

### 5. Support Vector Machine

The Support Vector Machine classifier was implemented using the **Radial Basis Function (RBF) kernel**.

The RBF kernel helps the model learn nonlinear decision boundaries between the seven bean classes.

Standardized features were used because SVM is sensitive to differences in feature scales.

---

## Model Evaluation Metrics

The following six evaluation metrics were calculated for every model.

### Accuracy

Accuracy represents the proportion of test observations that were classified correctly.

### AUC

Area Under the ROC Curve (AUC) measures the model's ability to distinguish between classes across different classification thresholds.

For this multiclass problem, One-vs-Rest AUC with weighted averaging was used.

### Precision

Precision measures how many observations predicted as a particular class actually belong to that class.

Weighted Precision was used because the target classes contain different numbers of records.

### Recall

Recall measures how many actual observations of each class were identified correctly.

Weighted Recall was used for this multiclass dataset.

### F1 Score

The F1 Score is the harmonic mean of Precision and Recall.

It provides a balanced performance measure, particularly when the class distribution is not perfectly uniform.

### Matthews Correlation Coefficient

Matthews Correlation Coefficient (MCC) evaluates the relationship between the actual and predicted classes.

MCC considers both correct and incorrect predictions and provides a useful balanced measure for multiclass and imbalanced classification.

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | **0.9934** | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| **Support Vector Machine** | **0.9210** | 0.9933 | **0.9214** | **0.9210** | **0.9211** | **0.9045** |

The metric values are rounded to four decimal places.

---

## Model Performance Observations

| ML Model Name | Observation About Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression performed very well, achieving an Accuracy of **0.9192** and an F1 Score of **0.9193**. It achieved the **highest AUC of 0.9934** among all five models, indicating excellent ability to distinguish between the seven bean classes. Its MCC of **0.9023** also demonstrates a strong relationship between the predicted and actual classes. The strong performance suggests that the standardized features provide considerable linear separation between the bean varieties. |
| **Decision Tree** | Decision Tree achieved an Accuracy of **0.8955** and an F1 Score of **0.8953**. Its performance was lower than Logistic Regression, KNN, and SVM. Its AUC of **0.9357** was the lowest AUC among the five models. A single unrestricted tree can create highly specific rules from the training data, which may reduce its ability to generalize to unseen observations. However, the model still classified almost 90% of the test observations correctly. |
| **KNN** | KNN achieved an Accuracy of **0.9155**, an F1 Score of **0.9157**, and an MCC of **0.8978**. Its performance was close to Logistic Regression and SVM, showing that observations belonging to the same bean class tend to be located near one another in the standardized feature space. Feature scaling was important because KNN calculates distances between observations. Its AUC of **0.9811** indicates strong multiclass discrimination. |
| **Naive Bayes** | Gaussian Naive Bayes produced the lowest overall performance, with an Accuracy of **0.7630**, an F1 Score of **0.7607**, and an MCC of **0.7143**. Its AUC remained relatively high at **0.9644**, indicating that its probability-based class ranking remained useful even though its final class predictions were less accurate. The weaker classification performance may be related to the conditional-independence and Gaussian-distribution assumptions of the model because several morphological bean measurements are related to one another. |
| **Support Vector Machine** | SVM achieved the **highest overall Accuracy of 0.9210, Precision of 0.9214, Recall of 0.9210, F1 Score of 0.9211, and MCC of 0.9045**. Its AUC of **0.9933** was also excellent and only 0.0001 below Logistic Regression. The RBF kernel enables SVM to learn nonlinear boundaries between similar bean varieties, while standardized features prevent measurements with large numerical ranges from dominating the model. |
| **Overall Winner for the Dataset** | **Support Vector Machine is the overall winner.** SVM achieved the best Accuracy, Precision, Recall, F1 Score, and MCC. Logistic Regression achieved a marginally higher AUC, but SVM provided the strongest overall balance across the six required evaluation metrics. Therefore, SVM was selected as the best-performing model for the Dry Bean Dataset. |

---

## Overall Conclusion

All five machine learning models successfully performed multiclass classification on the Dry Bean Dataset.

The results demonstrate that the morphological measurements of dry beans contain useful information for distinguishing between the seven bean varieties.

### Best Model: Support Vector Machine

Support Vector Machine produced the strongest overall results:

| Metric | SVM Score |
|---|---:|
| Accuracy | **0.9210** |
| AUC | **0.9933** |
| Precision | **0.9214** |
| Recall | **0.9210** |
| F1 Score | **0.9211** |
| MCC | **0.9045** |

Logistic Regression was a close second and achieved the highest AUC of **0.9934**.

KNN also performed strongly after feature standardization.

Decision Tree showed good but comparatively lower performance.

Gaussian Naive Bayes produced the weakest overall classification results, indicating that its assumptions were less appropriate for the relationships between the Dry Bean features.

Based on the complete set of evaluation metrics, **Support Vector Machine is the recommended model for classifying the bean varieties in this dataset.**

---
