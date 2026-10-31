# FCLCNN-LSTM for Telecom Customer Churn Risk Assessment

A from-scratch implementation of the hybrid deep-learning model in:

> C. Wang, C. Rao, F. Hu, X. Xiao, M. Goh. **Risk assessment of customer churn in telco using FCLCNN-LSTM model.** *Expert Systems with Applications* 248 (2024) 123352.

The model stacks a **Fully Connected Layer** (to homogenise heterogeneous customer features), a **2D CNN** (spatial feature extraction after reshaping the feature vector into a matrix), and an **LSTM** (sequential feature extraction), followed by dense layers for churn classification.

## What the notebook covers

| Step | Content |
|---|---|
| 1. Data | IBM Telco Customer Churn (7,043 customers, 21 columns, churn rate ≈ 26.5%); cleaning, label encoding, standardisation |
| 2. Feature selection | **Maj-LASSO** (majority-vote LASSO over balanced undersampled subsets) for imbalanced data: 30 → 22 features |
| 3. Model | FCL → reshape → 2D CNN → average pooling → LSTM → FC layers (TensorFlow/Keras) |
| 4. Training | Early stopping, learning-rate scheduling, class-imbalance handling, hyperparameter search with the **Whale Optimization Algorithm** |
| 5. Evaluation | Accuracy, precision, recall, F1, G-mean, AUC, confusion matrix, ROC and PR curves |
| 6. Comparison | Logistic Regression, Random Forest, SVM, XGBoost on the same split |

## Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.800 | 0.640 | 0.570 | 0.603 | 0.834 |
| Random Forest | 0.765 | 0.570 | 0.468 | 0.514 | 0.799 |
| SVM | 0.785 | 0.616 | 0.511 | 0.558 | 0.792 |
| XGBoost | 0.788 | 0.614 | 0.545 | 0.578 | 0.807 |
| **FCLCNN-LSTM** | 0.787 | 0.598 | **0.610** | **0.604** | 0.824 |

FCLCNN-LSTM gives the best recall and F1 (G-mean 0.721), i.e. it catches more of the customers who actually churn, which is the costly error in churn management. Logistic Regression remains a strong baseline on this small tabular dataset; the margin in F1 is small.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook fclcnn_lstm_churn.ipynb
```

The dataset is downloaded automatically from IBM's public copy (also on Kaggle: [blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)). The notebook also runs on Google Colab. Whale-optimisation settings range from a quick test (`n_whales=5, n_iterations=10`) to a full search (`n_whales=20, n_iterations=50`).

## Author

Hasti Tajik — M.Sc. Data Science, University of Science and Culture
