
# ============================================================
# CUSTOMER CHURN PREDICTION - COMPLETE MACHINE LEARNING PROJECT
# Algorithms: Decision Tree, Random Forest, Gradient Boosting
# Final Model: Soft Voting Ensemble
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

DATA_FILE = "Telco-Customer-Churn.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Download the CSV dataset "
        "and put it in the same folder as this Python file."
    )

df = pd.read_csv(DATA_FILE)

print("\n========== DATASET PREVIEW ==========")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nDataset information:")
df.info()

# ------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

print("\n========== DATA STATISTICS ==========")
print(df.describe(include="all").T)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# 3. DATA CLEANING
# ------------------------------------------------------------

# Convert TotalCharges into numeric values.
# Invalid/blank entries become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Fill missing TotalCharges with the median.
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove rows with missing target labels, if any
df = df.dropna(subset=["Churn"]).copy()

print("\n========== CLEANED DATA ==========")
print("Shape after cleaning:", df.shape)
print("Total missing values:", df.isnull().sum().sum())

# ------------------------------------------------------------
# 4. EDA VISUALIZATIONS
# ------------------------------------------------------------

# Graph 1: Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Graph 2: Contract type vs Churn
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Graph 3: Monthly charges vs Churn
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()

# Graph 4: Tenure vs Churn
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure vs Churn")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------

X = df.drop(
    columns=["Churn", "customerID"],
    errors="ignore"
)

# No = 0, Yes = 1
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isnull().any():
    raise ValueError(
        "Unexpected values found in the Churn target column."
    )

print("\n========== FEATURES AND TARGET ==========")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nTarget distribution:")
print(y.value_counts())

# Identify numerical and categorical columns
num_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

cat_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical columns:", num_cols)
print("\nCategorical columns:", cat_cols)

# ------------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ------------------------------------------------------------
# 7. PREPROCESSING
# ------------------------------------------------------------

# Numerical columns are passed through unchanged.
# Categorical columns are converted into numeric columns.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        )
    ]
)

# ------------------------------------------------------------
# 8. DECISION TREE MODEL
# ------------------------------------------------------------

dt_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        DecisionTreeClassifier(
            max_depth=6,
            random_state=42
        )
    )
])

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print(
    "\nDecision Tree Accuracy:",
    round(accuracy_score(y_test, dt_pred), 4)
)

# ------------------------------------------------------------
# 9. RANDOM FOREST MODEL
# ------------------------------------------------------------

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    )
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(
    "Random Forest Accuracy:",
    round(accuracy_score(y_test, rf_pred), 4)
)

# ------------------------------------------------------------
# 10. GRADIENT BOOSTING MODEL
# ------------------------------------------------------------

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    )
])

gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

print(
    "Gradient Boosting Accuracy:",
    round(accuracy_score(y_test, gb_pred), 4)
)

# ------------------------------------------------------------
# 11. FINAL ENSEMBLE MODEL
# ------------------------------------------------------------

final_model = VotingClassifier(
    estimators=[
        ("decision_tree", dt_model),
        ("random_forest", rf_model),
        ("gradient_boosting", gb_model)
    ],
    voting="soft"
)

final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

print(
    "Final Ensemble Accuracy:",
    round(accuracy_score(y_test, final_pred), 4)
)

# ------------------------------------------------------------
# 12. MODEL EVALUATION
# ------------------------------------------------------------

models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Final Ensemble": final_model
}

results = []

print("\n========== MODEL EVALUATION ==========")

for name, model in models.items():

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

# ------------------------------------------------------------
# 13. COMPARISON TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(
    results_df.round(4).to_string(index=False)
)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

# Compare models using F1 Score, which balances
# precision and recall for the positive (Churn) class.
best_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

best_name = best_row["Model"]
best_model = models[best_name]

print("\nBest model by test F1 Score:", best_name)
print("Best model F1 Score:", round(best_row["F1 Score"], 4))

# ------------------------------------------------------------
# 14. COMPARISON BAR CHART
# ------------------------------------------------------------

plot_df = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
]

plot_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Comparison of Machine Learning Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 15. CONFUSION MATRIX FOR FINAL ENSEMBLE
# ------------------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    final_pred,
    display_labels=["No Churn", "Churn"],
    cmap="Blues"
)

plt.title("Final Ensemble Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nFinal Ensemble Confusion Matrix:")
print(confusion_matrix(y_test, final_pred))

# ------------------------------------------------------------
# 16. ROC CURVE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random baseline"
)

plt.title("ROC Curves: Model Comparison")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 17. FEATURE IMPORTANCE (RANDOM FOREST)
# ------------------------------------------------------------

# Feature names after preprocessing
rf_preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

feature_names = rf_preprocessor.get_feature_names_out()
importances = rf_classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 15 IMPORTANT FEATURES ==========")
print(importance_df.head(15).to_string(index=False))

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 18. SAVE BEST MODEL
# ------------------------------------------------------------

joblib.dump(best_model, "churn_model.pkl")

print("\nBest model saved as churn_model.pkl")
print("Model type:", best_name)

# ------------------------------------------------------------
# 19. EXAMPLE CUSTOMER PREDICTIONS
# ------------------------------------------------------------

# These are illustrative customer records, not real people.
sample_customers = pd.DataFrame([
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": 151.65
    },
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.00,
        "TotalCharges": 3900.00
    }
])

# Match the training feature order exactly.
sample_customers = sample_customers[X.columns]

sample_predictions = best_model.predict(sample_customers)
sample_probabilities = best_model.predict_proba(
    sample_customers
)[:, 1]

prediction_df = pd.DataFrame({
    "Customer": [1, 2],
    "Churn Prediction": [
        "Yes" if p == 1 else "No"
        for p in sample_predictions
    ],
    "Churn Probability (%)": (
        sample_probabilities * 100
    ).round(2)
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(prediction_df.to_string(index=False))

prediction_df.to_csv(
    "sample_predictions.csv",
    index=False
)

print("\nProject completed successfully!")
print("Files created:")
print("- churn_model.pkl")
print("- model_comparison.csv")
print("- sample_predictions.csv")
