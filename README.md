# ML-5: Ensemble Methods

## Project Title
Comparison of Random Forest, Gradient Boosting, and XGBoost

## Objective
This project compares three ensemble learning methods on the same real-world
classification dataset:

- Random Forest
- Gradient Boosting
- XGBoost

The models use the same train/test split. Key hyperparameters are tuned and
the models are compared using Accuracy, Precision, Recall, F1 Score, feature
importance, and training time.

## Dataset
**Wisconsin Diagnostic Breast Cancer (WDBC) dataset**

The dataset contains diagnostic measurements computed from digitized images of
breast mass cell nuclei. The target variable is binary: malignant or benign.

## Project Structure

```text
ML5_Ensemble_Methods/
│
├── breast_cancer.csv
├── ensemble_comparison.py
├── baseline_results.csv
├── tuned_results.csv
├── top_15_feature_importance.csv
├── performance_comparison.png
├── feature_importance_comparison.png
├── training_time_comparison.png
├── conclusion.txt
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost

## How to Run

```bash
pip install -r requirements.txt
python ensemble_comparison.py
```

## Methodology

1. Load the WDBC dataset.
2. Use an identical 80/20 stratified train/test split.
3. Train baseline Random Forest, Gradient Boosting, and XGBoost models.
4. Tune important hyperparameters with 5-fold cross-validation.
5. Evaluate the final models on the same unseen test set.
6. Compare Accuracy, Precision, Recall, and F1 Score.
7. Compare feature importances.
8. Compare final-model training time.
9. Write a conclusion based on both performance and practical trade-offs.

## Hyperparameters Tuned

### Random Forest
- `n_estimators`
- `max_depth`
- `min_samples_split`

### Gradient Boosting
- `n_estimators`
- `learning_rate`
- `max_depth`

### XGBoost
- `n_estimators`
- `learning_rate`
- `max_depth`

## Fair Comparison
All three models use:
- the same dataset
- the same train/test split
- the same target
- the same test metrics

Hyperparameter tuning is performed only using the training set.

## Expected Output

The script generates:

- `baseline_results.csv`
- `tuned_results.csv`
- `top_15_feature_importance.csv`
- `performance_comparison.png`
- `feature_importance_comparison.png`
- `training_time_comparison.png`
- `conclusion.txt`

## Conclusion
The winning model should be selected based on test performance, especially
F1 Score, while also considering training time and the risk of overfitting.
Random Forest is based on bagging and variance reduction, while Gradient
Boosting and XGBoost use sequential error correction. XGBoost can often
achieve strong performance but may require more careful tuning.

## GitHub Submission

Create a **public** repository, upload all project files, and submit the
public GitHub repository URL in the assignment portal.
