import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)

from sklearn.pipeline import Pipeline


# ==========================================================
# 1. LOAD DATA
# ==========================================================

DATA_PATH = "HR_comma_sep.csv"

data = pd.read_csv(DATA_PATH)

print("=" * 60)
print("EMPLOYEE ATTRITION MODEL TRAINING")
print("=" * 60)

print("Dataset shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum())

print("\nTarget distribution:")
print(data["left"].value_counts())

print("\nTarget percentage:")
print(data["left"].value_counts(normalize=True) * 100)


# ==========================================================
# 2. REMOVE DUPLICATES
# ==========================================================

duplicates = data.duplicated().sum()

print("\nDuplicate rows:", duplicates)

if duplicates > 0:
    data = data.drop_duplicates().reset_index(drop=True)

print("Dataset shape after duplicate removal:", data.shape)


# ==========================================================
# 3. FEATURES AND TARGET
# ==========================================================

X = data.drop("left", axis=1)
y = data["left"]


# ==========================================================
# 4. IDENTIFY FEATURES
# ==========================================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ==========================================================
# 5. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ==========================================================
# 6. PREPROCESSING
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================================
# 7. MODELS
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ==========================================================
# 8. CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = {}

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_results[name] = scores

    print(f"\n{name}")
    print("AUC Scores:", np.round(scores, 4))
    print("Average AUC:", round(scores.mean(), 4))
    print("Standard Deviation:", round(scores.std(), 4))


# ==========================================================
# 9. FINAL RANDOM FOREST MODEL
# ==========================================================

final_model = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


# ==========================================================
# 10. COMPLETE FINAL PIPELINE
# ==========================================================

final_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", final_model)
    ]
)


# ==========================================================
# 11. TRAIN FINAL MODEL
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 60)

final_pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ==========================================================
# 12. TEST PREDICTIONS
# ==========================================================

y_pred = final_pipeline.predict(X_test)

y_prob = final_pipeline.predict_proba(X_test)[:, 1]


# ==========================================================
# 13. ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==========================================================
# 14. CLASSIFICATION REPORT
# ==========================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ==========================================================
# 15. ROC-AUC
# ==========================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print(f"ROC-AUC: {roc_auc:.4f}")


# ==========================================================
# 16. PRECISION-RECALL AUC
# ==========================================================

precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

pr_auc = auc(
    recall,
    precision
)

print(f"PR-AUC: {pr_auc:.4f}")


# ==========================================================
# 17. SAVE COMPLETE MODEL
# ==========================================================

MODEL_PATH = "model.pkl"

joblib.dump(
    final_pipeline,
    MODEL_PATH,
    compress=3
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Saved file: {MODEL_PATH}")


# ==========================================================
# 18. MODEL FILE SIZE
# ==========================================================

import os

model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print(f"Model size: {model_size_mb:.2f} MB")


# ==========================================================
# 19. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Model: Random Forest")
print("Number of trees: 100")
print("SMOTE: Not used")
print("Class Weight: Not used")

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test ROC-AUC: {roc_auc:.4f}")
print(f"Test PR-AUC: {pr_auc:.4f}")

print("\nCross-validation results:")

for name, scores in cv_results.items():

    print(
        f"{name}: "
        f"{scores.mean():.4f} "
        f"(+/- {scores.std():.4f})"
    )

print(f"\nFinal model size: {model_size_mb:.2f} MB")

print("\nTraining completed successfully.")
print("Deployment file: model.pkl")