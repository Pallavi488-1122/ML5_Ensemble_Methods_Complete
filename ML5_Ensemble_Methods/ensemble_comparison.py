import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

RANDOM_STATE = 42

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("breast_cancer.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# -----------------------------
# Baseline models
# -----------------------------
baseline_models = {
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )
}

baseline_results = []

for name, model in baseline_models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    pred = model.predict(X_test)

    baseline_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "Training Time (sec)": elapsed
    })

baseline_df = pd.DataFrame(baseline_results)
baseline_df.to_csv("baseline_results.csv", index=False)

# -----------------------------
# Hyperparameter tuning
# -----------------------------
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    }
}

base_tuning_models = {
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )
}

best_models = {}
tuned_results = []

for name in base_tuning_models:
    grid = GridSearchCV(
        estimator=base_tuning_models[name],
        param_grid=param_grids[name],
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    tuning_start = time.perf_counter()
    grid.fit(X_train, y_train)
    tuning_time = time.perf_counter() - tuning_start

    best_model = grid.best_estimator_

    # Measure final selected model's training time on the same training data.
    train_start = time.perf_counter()
    best_model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    pred = best_model.predict(X_test)

    best_models[name] = best_model

    tuned_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "Training Time (sec)": train_time,
        "Tuning Time (sec)": tuning_time,
        "Best Parameters": str(grid.best_params_)
    })

    print("\n" + "=" * 70)
    print(name)
    print("Best Parameters:", grid.best_params_)
    print(classification_report(y_test, pred))

tuned_df = pd.DataFrame(tuned_results)
tuned_df.to_csv("tuned_results.csv", index=False)

# -----------------------------
# Performance comparison plot
# -----------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

ax = tuned_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)
ax.set_title("Performance Comparison of Ensemble Models")
ax.set_ylabel("Score")
ax.set_ylim(0.80, 1.01)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=200)
plt.close()

# -----------------------------
# Training time comparison
# -----------------------------
ax = tuned_df.set_index("Model")["Training Time (sec)"].plot(
    kind="bar",
    figsize=(9, 5)
)
ax.set_title("Training Time Comparison")
ax.set_ylabel("Training Time (seconds)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("training_time_comparison.png", dpi=200)
plt.close()

# -----------------------------
# Feature importance comparison
# -----------------------------
importance_df = pd.DataFrame(index=X.columns)

for name, model in best_models.items():
    importance_df[name] = model.feature_importances_

# Top 15 features based on average importance
importance_df["Average Importance"] = importance_df.mean(axis=1)
top_features = importance_df.sort_values(
    "Average Importance", ascending=False
).head(15).drop(columns=["Average Importance"])

top_features.to_csv("top_15_feature_importance.csv")

ax = top_features.sort_values("Average Importance" if "Average Importance" in top_features.columns else top_features.columns[0])

# Plot all three models side-by-side for the same top 15 features.
plot_df = top_features.copy()
plot_df = plot_df.loc[
    plot_df.mean(axis=1).sort_values(ascending=False).head(15).index
]

plot_df.plot(kind="bar", figsize=(15, 7))
plt.title("Top 15 Feature Importances: Random Forest vs Gradient Boosting vs XGBoost")
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=200)
plt.close()

# -----------------------------
# Save final summary
# -----------------------------
winner = tuned_df.loc[tuned_df["F1 Score"].idxmax(), "Model"]

with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(
        "Ensemble Methods - Conclusion\n"
        "============================\n\n"
        f"Best model based on test F1 Score: {winner}\n\n"
        "The three ensemble methods were trained using the same Wisconsin "
        "Diagnostic Breast Cancer dataset, the same 80/20 stratified split, "
        "and comparable evaluation metrics. Hyperparameters were tuned using "
        "5-fold cross-validation on the training set only.\n\n"
        "Random Forest uses bagging and averages many decision trees, which "
        "mainly reduces variance. Gradient Boosting and XGBoost build trees "
        "sequentially, allowing later trees to correct previous errors. "
        "XGBoost also includes additional regularization and optimized "
        "implementation details, which can improve predictive performance "
        "but may increase tuning complexity.\n\n"
        "The final choice should consider not only F1/accuracy but also "
        "training time, model complexity, and the train-test performance gap."
    )

print("\nFinal tuned comparison:")
print(tuned_df.to_string(index=False))
print("\nProject completed. Check the generated CSV files and PNG plots.")
