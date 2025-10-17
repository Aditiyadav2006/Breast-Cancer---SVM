# 🎗️ Breast Cancer Prediction: Support Vector Machine (SVM) Classification

## Project Overview

This project implements a **Support Vector Machine (SVM)** classifier to predict whether a breast mass is **Malignant (cancerous)** or **Benign (non-cancerous)** based on features extracted from digitized images of fine needle aspirates (FNA).

The goal is to build a highly accurate and reliable model, which is crucial in the medical domain. The project includes essential machine learning steps such as data preparation, scaling, model training, and rigorous evaluation.

## Key Files

| File Name | Description |
| :--- | :--- |
| `breast_cancer_svm.ipynb` | The main Jupyter Notebook containing all the steps: data loading, preprocessing (scaling), SVM model training, and performance evaluation. |
| `data.csv` | The dataset containing features calculated from the cell nuclei (e.g., radius, texture, perimeter, area, etc.) and the target variable (`diagnosis`). |
| `classification_report_breast_cancer.txt` | Text file containing the detailed precision, recall, and F1-score for each class, as well as the overall metrics. |
| `confusion_matirx_breast_cancer.png` | Visualization of the model's confusion matrix, showing the counts of correct and incorrect predictions. |
| `README.md` | This overview file. |

## Methodology

### 1. Data Preprocessing
The dataset features were inspected, and crucial preprocessing steps were performed:

* **Handling Categorical Data:** The `diagnosis` column (M/B) was converted into numerical format (Malignant=1, Benign=0).
* **Feature Scaling:** **StandardScaler** was applied to the numerical features. This is a critical step for SVMs, as they are distance-based algorithms and perform best when features are on the same scale.

### 2. Support Vector Machine (SVM) Model
An SVM model with an appropriate kernel (often RBF or linear) was trained for binary classification. SVM is known for its effectiveness in high-dimensional spaces and cases where the data is not perfectly separable.

### 3. Model Evaluation

The model was evaluated on a held-out test set. Here are the exceptional results:

| Metric | Result |
| :--- | :--- |
| **Accuracy** | $\mathbf{0.98}$ |
| **Precision (Weighted Avg)** | $\mathbf{0.98}$ |
| **Recall (Weighted Avg)** | $\mathbf{0.98}$ |
| **F1-Score (Weighted Avg)** | $\mathbf{0.98}$ |

## Key Findings and Analysis

The model performance is virtually perfect, which is a common and excellent result for the Wisconsin Breast Cancer Diagnostic dataset when using strong classifiers like SVM:

1.  **High Accuracy ($\mathbf{98\%}$):** The model correctly predicted the diagnosis ($\mathbf{Malignant}$ or $\mathbf{Benign}$) for $\mathbf{98\%}$ of the test samples.
2.  **Balanced Performance (F1-Score $\mathbf{0.98}$):** The high and equal Precision, Recall, and F1-Scores across all classes confirm that the model is extremely robust and does not favor one diagnosis over the other.
3.  **Critical Metric Performance (Focus on Malignant):**
    * **High Precision for Malignant:** This minimizes **False Positives** (predicting Malignant when it is actually Benign), which is vital to avoid unnecessary stress and invasive follow-up procedures for patients.
    * **High Recall for Malignant:** This minimizes **False Negatives** (predicting Benign when it is actually Malignant), which is the most critical error in a medical diagnosis, as it delays necessary life-saving treatment. The $\mathbf{98\%}$ recall indicates the model is highly successful at finding true cancer cases.
4.  **Confirmation via Confusion Matrix:** The `confusion_matirx_breast_cancer.png` visually confirms that the count of misclassifications (False Positives and False Negatives) is extremely low.

## Technologies and Libraries

* **Python 3.x**
* **VS code**
* `pandas` & `numpy` (for data handling)
* `scikit-learn` (for `SVC` classifier, `StandardScaler`, `train_test_split`, and metrics)
* `matplotlib` & `seaborn` (for visualization)
You can also view the notebook directly on GitHub or platforms like **Google Colab** without needing a local setup.

***

## 👩‍💻 Author
**Aditi K**  
CSE (AI & ML) | Breast Cancer| SVM
