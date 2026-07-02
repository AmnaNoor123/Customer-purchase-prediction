# Predicting Online Shopper Purchase Intention

A comparative machine learning project that predicts whether a website visitor's session will end in a purchase, using three classification algorithms: **K-Nearest Neighbors (KNN)**, **Logistic Regression**, and **Decision Tree**.

> 📚 This is a **university team project**, submitted as part of a Machine Learning course assignment. Shared here for academic and portfolio purposes.

**Team Members:**
- Name 1 — Amna Noor(Leader)
- Name 2 — Ifra Ismail
- Name 3 — Iqra Yaar

## Dataset

[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) — UCI Machine Learning Repository.

- 12,330 website sessions collected over a one-year period
- 17 behavioural / technical features + `Revenue` target (True = purchase made)
- Class distribution: 84.5% No Purchase / 15.5% Purchase (imbalanced)

## Project Pipeline

Each notebook follows the same consistent, leakage-free pipeline:

1. Data cleaning (duplicate removal, missing value check)
2. Label encoding of categorical features
3. Train/test split (80/20, `random_state=42`)
4. Feature selection — `SelectKBest` with `mutual_info_classif` (top 10 features), fit **only on training data**
5. Feature scaling (where required)
6. Class imbalance handling (SMOTE for KNN, `class_weight='balanced'` for Logistic Regression & Decision Tree)
7. Model training and evaluation

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| K-Nearest Neighbors | 0.8406 | 0.4753 | 0.7182 | 0.5721 |
| **Logistic Regression** | **0.8726** | **0.5539** | 0.7238 | **0.6275** |
| Decision Tree | 0.8533 | 0.5034 | **0.8177** | 0.6232 |

*(Metrics reported for the positive/minority class — "Purchase".)*

**Logistic Regression** achieved the best overall balance of accuracy, precision, and F1 score. **Decision Tree** achieved the highest recall, making it preferable when catching as many actual buyers as possible matters more than avoiding false positives (e.g. marketing retargeting).

Across all three models, **PageValues** emerged as by far the strongest predictor of purchase intent, followed by **Month**, **ProductRelated(_Duration)**, and **ExitRates/BounceRates**.

## Files

| File | Description |
|---|---|
| `KNN Implementation.ipynb` | KNN model — feature selection, SMOTE oversampling, best-K search, evaluation |
| `Logistic Regression Implementation.ipynb` | Logistic Regression model — feature selection, class-weighted training, coefficient analysis |
| `Decision Tree Implementation.ipynb` | Decision Tree model — feature selection, class-weighted training, feature importance, tree visualization |
| `ML Project Documentation.docx` | Full written report with methodology, results, graphs, and model comparison |

## How to Run

1. Download the dataset CSV from the UCI link above (or a mirror such as [this GitHub copy](https://raw.githubusercontent.com/sharmaroshan/Online-Shoppers-Purchasing-Intention/master/online_shoppers_intention.csv))
2. Update the `pd.read_csv(...)` path in each notebook to point to the CSV
3. Run all cells top to bottom (`Run All`)

## Limitations

- Dataset is significantly imbalanced; minority-class metrics carry more uncertainty than majority-class metrics
- Only 10 of 17 available features were used per model
- No cross-validation was performed — results are from a single 80/20 split

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT) — free to use, modify, and share with attribution.

The dataset itself is not authored by this project; it belongs to the UCI Machine Learning Repository and is used here strictly for academic purposes. See the [dataset page](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) for its original terms of use.
