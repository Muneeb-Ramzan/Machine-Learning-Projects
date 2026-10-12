# ============================================================
# HEART DISEASE PREDICTION - COMPLETE MACHINE LEARNING PROJECT
# Based on the uploaded project_heartdiease.ipynb notebook
# Dataset required: heart_disease_uci.csv
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

RANDOM_STATE = 42


# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------
print("\nSTEP 1: Loading dataset...")

CSV_FILE = "heart_disease_uci.csv"

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"\nCould not find '{CSV_FILE}'. Put the CSV file in the same "
        "folder as this Python script, or update CSV_FILE with its full path."
    )

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape (rows, columns):", df.shape)

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all").T)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())


# ------------------------------------------------------------
# STEP 2: CHECK REQUIRED TARGET COLUMN AND CREATE BINARY TARGET
# ------------------------------------------------------------
print("\nSTEP 2: Creating target column...")

if "num" not in df.columns:
    raise ValueError(
        "The CSV must contain a 'num' column. Check that you are using "
        "the correct Heart Disease UCI dataset."
    )

print("\nOriginal 'num' value counts:")
print(df["num"].value_counts(dropna=False).sort_index())

# The notebook converts num=0 to 0 and num>0 to 1.
df = df.dropna(subset=["num"]).copy()
df["target"] = (df["num"] > 0).astype(int)

print("\nBinary target counts (0 = no disease indicated, 1 = disease indicated):")
print(df["target"].value_counts())

plt.figure(figsize=(8, 5))
df["num"].value_counts().sort_index().sort_index().plot(kind="bar")
plt.title("Distribution of Original num Values")
plt.xlabel("num")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="target")
plt.title("Distribution of Binary Target")
plt.xlabel("Target (0 = no disease indicated, 1 = disease indicated)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 3: EXPLORE FEATURES AND PLOT GRAPHS
# ------------------------------------------------------------
print("\nSTEP 3: Exploring data...")

numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

print("\nNumeric columns:")
print(list(numeric_cols))

print("\nCategorical columns:")
print(list(categorical_cols))

# Histograms for numeric columns
df[numeric_cols].hist(figsize=(15, 12), bins=20)
plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.show()

# Boxplots for numeric columns
plt.figure(figsize=(15, 8))
df[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplots of Numerical Features")
plt.tight_layout()
plt.show()

# Correlation heatmap (numeric columns only)
plt.figure(figsize=(12, 8))
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Age and cholesterol vs target, if available
if "age" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="target", y="age")
    plt.title("Age vs Target")
    plt.tight_layout()
    plt.show()

if "chol" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="target", y="chol")
    plt.title("Cholesterol vs Target")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# STEP 4: SEPARATE FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------
print("\nSTEP 4: Separating features and target...")

# Remove 'num' because target was created from it; this prevents leakage.
X = df.drop(columns=["num", "target"])
y = df["target"]

print("X shape:", X.shape)
print("y shape:", y.shape)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# ------------------------------------------------------------
# STEP 5: TRAIN-TEST SPLIT
# ------------------------------------------------------------
print("\nSTEP 5: Splitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ------------------------------------------------------------
# STEP 6: PREPROCESS MISSING VALUES AND CATEGORICAL FEATURES
# Fit imputers only on training data to avoid data leakage.
# ------------------------------------------------------------
print("\nSTEP 6: Preprocessing data...")

# Numeric missing values -> median
if numeric_features:
    num_imputer = SimpleImputer(strategy="median")
    X_train_num = pd.DataFrame(
        num_imputer.fit_transform(X_train[numeric_features]),
        columns=numeric_features,
        index=X_train.index,
    )
    X_test_num = pd.DataFrame(
        num_imputer.transform(X_test[numeric_features]),
        columns=numeric_features,
        index=X_test.index,
    )
else:
    X_train_num = pd.DataFrame(index=X_train.index)
    X_test_num = pd.DataFrame(index=X_test.index)

# Categorical missing values -> most frequent value
if categorical_features:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_cat = pd.DataFrame(
        cat_imputer.fit_transform(X_train[categorical_features]),
        columns=categorical_features,
        index=X_train.index,
    )
    X_test_cat = pd.DataFrame(
        cat_imputer.transform(X_test[categorical_features]),
        columns=categorical_features,
        index=X_test.index,
    )

    # Convert text categories into numeric indicator columns
    X_train_cat = pd.get_dummies(
        X_train_cat, columns=categorical_features, drop_first=True
    )
    X_test_cat = pd.get_dummies(
        X_test_cat, columns=categorical_features, drop_first=True
    )

    # Make test columns match training columns exactly
    X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)
else:
    X_train_cat = pd.DataFrame(index=X_train.index)
    X_test_cat = pd.DataFrame(index=X_test.index)

X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)

# Ensure identical column order
X_test_processed = X_test_processed.reindex(
    columns=X_train_processed.columns, fill_value=0
)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
print(
    "Missing values in processed training data:",
    int(X_train_processed.isnull().sum().sum()),
)
print(
    "Missing values in processed testing data:",
    int(X_test_processed.isnull().sum().sum()),
)


# ------------------------------------------------------------
# STEP 7: SCALE FEATURES (used for Gaussian Naive Bayes here)
# ------------------------------------------------------------
print("\nSTEP 7: Scaling features for Gaussian Naive Bayes...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)


# ------------------------------------------------------------
# STEP 8: TRAIN MODELS
# ------------------------------------------------------------
print("\nSTEP 8: Training models...")

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_model.fit(X_train_processed, y_train)
y_pred_dt = dt_model.predict(X_test_processed)
y_prob_dt = dt_model.predict_proba(X_test_processed)[:, 1]

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X_train_processed, y_train)
y_pred_rf = rf_model.predict(X_test_processed)
y_prob_rf = rf_model.predict_proba(X_test_processed)[:, 1]

# Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)
y_prob_nb = nb_model.predict_proba(X_test_scaled)[:, 1]

print("All three models have been trained.")


# ------------------------------------------------------------
# STEP 9: EVALUATE MODELS
# ------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["No disease indicated", "Disease indicated"],
            zero_division=0,
        )
    )


evaluate_model("DECISION TREE", y_test, y_pred_dt, y_prob_dt)
evaluate_model("RANDOM FOREST", y_test, y_pred_rf, y_prob_rf)
evaluate_model("GAUSSIAN NAIVE BAYES", y_test, y_pred_nb, y_prob_nb)


# ------------------------------------------------------------
# STEP 10: CONFUSION MATRICES
# ------------------------------------------------------------
for name, y_pred, cmap in [
    ("Decision Tree", y_pred_dt, "Blues"),
    ("Random Forest", y_pred_rf, "Greens"),
    ("Gaussian Naive Bayes", y_pred_nb, "Oranges"),
]:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap=cmap,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# STEP 11: VISUALIZE DECISION TREE
# ------------------------------------------------------------
plt.figure(figsize=(24, 12))
plot_tree(
    dt_model,
    feature_names=list(X_train_processed.columns),
    class_names=["No Disease", "Disease"],
    filled=True,
    max_depth=3,
    fontsize=8,
)
plt.title("Decision Tree (first 3 levels)")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 12: RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train_processed.columns,
).sort_values(ascending=False)

plt.figure(figsize=(12, 7))
feature_importance.head(15).sort_values().plot(kind="barh")
plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 13: COMPARE THE THREE MODELS
# ------------------------------------------------------------
results = pd.DataFrame(
    [
        {
            "Model": "Decision Tree",
            "Accuracy": accuracy_score(y_test, y_pred_dt),
            "Precision": precision_score(y_test, y_pred_dt, zero_division=0),
            "Recall": recall_score(y_test, y_pred_dt, zero_division=0),
            "F1": f1_score(y_test, y_pred_dt, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob_dt),
            "PR-AUC": average_precision_score(y_test, y_prob_dt),
        },
        {
            "Model": "Random Forest",
            "Accuracy": accuracy_score(y_test, y_pred_rf),
            "Precision": precision_score(y_test, y_pred_rf, zero_division=0),
            "Recall": recall_score(y_test, y_pred_rf, zero_division=0),
            "F1": f1_score(y_test, y_pred_rf, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob_rf),
            "PR-AUC": average_precision_score(y_test, y_prob_rf),
        },
        {
            "Model": "Gaussian Naive Bayes",
            "Accuracy": accuracy_score(y_test, y_pred_nb),
            "Precision": precision_score(y_test, y_pred_nb, zero_division=0),
            "Recall": recall_score(y_test, y_pred_nb, zero_division=0),
            "F1": f1_score(y_test, y_pred_nb, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob_nb),
            "PR-AUC": average_precision_score(y_test, y_prob_nb),
        },
    ]
)

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
].plot(kind="bar", figsize=(13, 7))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 14: ROC CURVE COMPARISON
# ------------------------------------------------------------
plt.figure(figsize=(9, 7))
for name, probabilities in [
    ("Decision Tree", y_prob_dt),
    ("Random Forest", y_prob_rf),
    ("Naive Bayes", y_prob_nb),
]:
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_value = roc_auc_score(y_test, probabilities)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 15: PRECISION-RECALL CURVE COMPARISON
# ------------------------------------------------------------
plt.figure(figsize=(9, 7))
for name, probabilities in [
    ("Decision Tree", y_prob_dt),
    ("Random Forest", y_prob_rf),
    ("Naive Bayes", y_prob_nb),
]:
    precision_values, recall_values, _ = precision_recall_curve(
        y_test, probabilities
    )
    ap_value = average_precision_score(y_test, probabilities)
    plt.plot(
        recall_values,
        precision_values,
        label=f"{name} (AP = {ap_value:.3f})",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 16: SOFT-VOTING ENSEMBLE
# Average the positive-class probabilities of the three models.
# ------------------------------------------------------------
ensemble_prob = (y_prob_dt + y_prob_rf + y_prob_nb) / 3
ensemble_pred = (ensemble_prob >= 0.5).astype(int)

print("\nSOFT VOTING ENSEMBLE")
print(f"Accuracy : {accuracy_score(y_test, ensemble_pred):.4f}")
print(
    f"Precision: {precision_score(y_test, ensemble_pred, zero_division=0):.4f}"
)
print(f"Recall   : {recall_score(y_test, ensemble_pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, ensemble_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, ensemble_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, ensemble_prob):.4f}")
print("\nEnsemble classification report:")
print(
    classification_report(
        y_test,
        ensemble_pred,
        target_names=["No disease indicated", "Disease indicated"],
        zero_division=0,
    )
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, ensemble_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
)
plt.title("Soft Voting Ensemble - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 17: FINAL COMPARISON INCLUDING ENSEMBLE
# ------------------------------------------------------------
comparison = pd.concat(
    [
        results,
        pd.DataFrame(
            [
                {
                    "Model": "Soft Voting Ensemble",
                    "Accuracy": accuracy_score(y_test, ensemble_pred),
                    "Precision": precision_score(
                        y_test, ensemble_pred, zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test, ensemble_pred, zero_division=0
                    ),
                    "F1": f1_score(y_test, ensemble_pred, zero_division=0),
                    "ROC-AUC": roc_auc_score(y_test, ensemble_prob),
                    "PR-AUC": average_precision_score(y_test, ensemble_prob),
                }
            ]
        ),
    ],
    ignore_index=True,
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
print(comparison.to_string(index=False))

comparison.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
].plot(kind="bar", figsize=(14, 7))
plt.title("Three Models vs Soft Voting Ensemble")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

best_auc = comparison.loc[comparison["ROC-AUC"].idxmax()]
best_accuracy = comparison.loc[comparison["Accuracy"].idxmax()]

print("\nHighest ROC-AUC model:")
print(best_auc.to_string())

print("\nHighest accuracy model:")
print(best_accuracy.to_string())

print(
    "\nProject finished. This is an educational model, not a medical diagnosis tool."
)
