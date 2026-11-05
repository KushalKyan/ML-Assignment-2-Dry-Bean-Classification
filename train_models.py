# ============================================================
# ML Assignment 2
# Dry Bean Classification
#
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. K-Nearest Neighbors (KNN)
# 4. Naive Bayes
# 5. Support Vector Machine (SVM)
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = "Dry_Bean_Dataset.xlsx"
MODEL_DIR = "model"

# Create the model directory if it does not already exist.
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

print("=" * 60)
print("DRY BEAN CLASSIFICATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_excel(DATASET_PATH)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 3. BASIC DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

# Check missing values.
missing_values = df.isnull().sum().sum()

print("\nTotal missing values:")
print(missing_values)

# Check duplicate rows.
duplicate_rows = df.duplicated().sum()

print("\nDuplicate rows:")
print(duplicate_rows)

# Remove duplicate rows if any are found.
if duplicate_rows > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nRemoved {duplicate_rows} duplicate rows.")
    print("Dataset shape after duplicate removal:", df.shape)
else:
    print("\nNo duplicate rows found.")


# ============================================================
# 4. DISPLAY TARGET CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(df["Class"].value_counts())


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

# X contains the 16 input features.
X = df.drop(columns=["Class"])

# y contains the bean class we want to predict.
y = df["Class"]

print("\nFeature data shape:", X.shape)
print("Target data shape:", y.shape)


# ============================================================
# 6. ENCODE THE TARGET LABELS
# ============================================================

# Convert textual classes such as BARBUNYA, BOMBAY, CALI, etc.
# into numeric labels that can be used consistently by the models.

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClass label mapping:")

for index, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {index}")


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

# 80% of the data is used for training.
# 20% is kept aside for testing.
#
# stratify=y_encoded maintains approximately the same class
# distribution in both the training and testing datasets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training features:", X_train.shape)
print("Testing features :", X_test.shape)
print("Training labels  :", y_train.shape)
print("Testing labels   :", y_test.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

# StandardScaler is fitted ONLY on training data.
# This prevents information from the test dataset leaking
# into the model training process.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. DEFINE THE FIVE REQUIRED MODELS
# ============================================================

models = {

    # Scaling will be used for Logistic Regression.
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "scaled": True,
        "filename": "logistic_regression.pkl"
    },

    # Decision Trees do not require feature scaling.
    "Decision Tree": {
        "model": DecisionTreeClassifier(
            random_state=42
        ),
        "scaled": False,
        "filename": "decision_tree.pkl"
    },

    # KNN is distance based, so scaling is important.
    "KNN": {
        "model": KNeighborsClassifier(
            n_neighbors=5
        ),
        "scaled": True,
        "filename": "knn.pkl"
    },

    # Gaussian Naive Bayes can work directly with the
    # original numerical features.
    "Naive Bayes": {
        "model": GaussianNB(),
        "scaled": False,
        "filename": "naive_bayes.pkl"
    },

    # probability=True is required because we need
    # class probabilities for ROC-AUC calculation.
    "SVM": {
        "model": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "scaled": True,
        "filename": "svm.pkl"
    }
}


# ============================================================
# 10. FUNCTION TO CALCULATE EVALUATION METRICS
# ============================================================

def evaluate_model(model, X_eval, y_true):
    """
    Evaluate a trained classification model.

    Returns:
        Accuracy
        AUC
        Precision
        Recall
        F1 Score
        MCC
    """

    # Predict class labels.
    y_pred = model.predict(X_eval)

    # Predict class probabilities.
    # These are needed for multiclass ROC-AUC.
    y_prob = model.predict_proba(X_eval)

    # Accuracy
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # Weighted precision is used because this is
    # a multiclass classification problem.
    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Weighted recall.
    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Weighted F1 score.
    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Matthews Correlation Coefficient.
    # MCC also supports multiclass classification.
    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    # Multiclass ROC-AUC using One-vs-Rest.
    auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )

    return accuracy, auc, precision, recall, f1, mcc


# ============================================================
# 11. TRAIN AND EVALUATE ALL MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

results = []

for model_name, config in models.items():

    print(f"\nTraining {model_name}...")

    model = config["model"]

    # Select scaled or unscaled data depending on the model.
    if config["scaled"]:
        training_data = X_train_scaled
        testing_data = X_test_scaled
    else:
        training_data = X_train
        testing_data = X_test

    # Train the model.
    model.fit(
        training_data,
        y_train
    )

    # Evaluate the trained model.
    accuracy, auc, precision, recall, f1, mcc = evaluate_model(
        model,
        testing_data,
        y_test
    )

    # Store results for comparison.
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # Save the trained model.
    model_path = os.path.join(
        MODEL_DIR,
        config["filename"]
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"{model_name} trained successfully.")
    print(f"Model saved to: {model_path}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")


# ============================================================
# 12. CREATE MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. SAVE MODEL RESULTS
# ============================================================

# Saving the metrics is useful when building the Streamlit app
# and when preparing the README/report.

results_df.to_csv(
    "model_results.csv",
    index=False
)

print("\nModel results saved as model_results.csv")


# ============================================================
# 14. SAVE THE SCALER
# ============================================================

scaler_path = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)

print("Scaler saved successfully.")


# ============================================================
# 15. SAVE THE LABEL ENCODER
# ============================================================

label_encoder_path = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

joblib.dump(
    label_encoder,
    label_encoder_path
)

print("Label encoder saved successfully.")


# ============================================================
# 16. SAVE FEATURE NAMES
# ============================================================

# Keeping the feature names allows Streamlit to confirm
# that uploaded data contains the correct columns.

feature_names = X.columns.tolist()

feature_names_path = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)

joblib.dump(
    feature_names,
    feature_names_path
)

print("Feature names saved successfully.")


# ============================================================
# 17. CREATE TEST_DATA.CSV
# ============================================================

# Make a copy so that we don't modify X_test.
test_data = X_test.copy()

# Convert the encoded target labels back to original bean
# class names before saving the test CSV.
test_data["Class"] = label_encoder.inverse_transform(y_test)

# Save test data.
test_data.to_csv(
    "test_data.csv",
    index=False
)

print("Test data saved as test_data.csv")


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nCreated files:")

print("""
model/
    logistic_regression.pkl
    decision_tree.pkl
    knn.pkl
    naive_bayes.pkl
    svm.pkl
    scaler.pkl
    label_encoder.pkl
    feature_names.pkl

model_results.csv
test_data.csv
""")

print("All five models are ready for the Streamlit application.")