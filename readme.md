# Customer Churn Predictive Analysis for Lloyds Banking Group

**Project Status:** Complete  
**Date:** August 3, 2025  
**Author:** Costas Pinto, Data Science Team

---
<img width="640" height="480" alt="confusion_matrix_phase2" src="https://github.com/user-attachments/assets/beda3222-d187-41e6-854e-12bc3dbeda3d" />
<img width="1000" height="600" alt="feature_importance_phase2" src="https://github.com/user-attachments/assets/3d071817-6d8f-4bc2-a483-b99bcecfb17e" />
<img width="800" height="500" alt="prediction_histogram_phase2" src="https://github.com/user-attachments/assets/fe44c786-f731-4b0b-b53f-892d4ffc9d6a" />
<img width="640" height="480" alt="roc_curve_phase2" src="https://github.com/user-attachments/assets/864c1778-5d5e-44c1-a241-13c8a08bf043" />


## 1. Project Overview

This repository contains the end-to-end data science pipeline developed to predict customer churn at **SmartBank**, a subsidiary of Lloyds Banking Group. The primary objective is to **proactively identify high-risk customers** using demographic, transactional, and behavioral data.

The final model's insights aim to support **targeted customer retention strategies**, reducing churn-related revenue loss and improving loyalty.

The project was executed in two main phases:

- **Phase 1: Data Foundation & EDA**  
  - Data integration from multiple sources  
  - Advanced feature engineering  
  - Exploratory Data Analysis (EDA) for initial insights

- **Phase 2: Predictive Modeling**  
  - Training and evaluation of multiple machine learning models  
  - Hyperparameter tuning and class imbalance handling  
  - Selection of the best churn prediction model

---

## 2. Project Structure

```

LOYDS DATA SCIENCE/
│
├── datasets/
│   └── Customer\_Churn\_Data\_Large.xlsx     # Raw input data
│
├── logs/                                  # Script execution logs
├── models/                                # Trained model artifacts
├── outputs/                               # Processed datasets and reports
├── plots/                                 # EDA and model evaluation visuals
├── reports/                               # PDF reports for each phase
│
├── eda.py                                 # EDA script
├── preprocess.py                          # Data cleaning & feature engineering
├── train.py                               # Model training & evaluation
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation

````

---

## 3. Setup and Installation

### Prerequisites

- Python 3.8+
- `pip` and `venv`

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/lloyds-churn-prediction.git
   cd lloyds-churn-prediction


2. **Create and Activate Virtual Environment**

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Place the Dataset**

   * Ensure `Customer_Churn_Data_Large.xlsx` is inside the `datasets/` folder.

---

## 4. Execution Workflow

### Step 1: Data Preprocessing

Run `preprocess.py` to clean and transform raw data into a model-ready format:

```bash
python preprocess.py
```

* Input: `datasets/Customer_Churn_Data_Large.xlsx`
* Output: `outputs/prepared_churn_dataset.csv`

---

### Step 2: Model Training and Evaluation

Run `train.py` to train and evaluate machine learning models:

```bash
python train.py
```

* Uses the prepared dataset
* Trains Logistic Regression, Random Forest, Gradient Boosting, and SVM
* Handles class imbalance using SMOTE
* Outputs:

  * Evaluation metrics and comparison table (`outputs/`)
  * Best model (`models/`)
  * Visualizations (`plots/`)

---

### (Optional) Step 3: Exploratory Data Analysis

Run `eda.py` to generate standalone exploratory analysis visuals:

```bash
python eda.py
```

---

## 5. Results and Recommendations

### Final Model Performance

* **Best Model:** Random Forest
* **Test AUC:** 0.5504

### Key Findings

Despite a robust pipeline and thorough feature engineering, **the AUC score is low**, indicating the current data lacks strong predictive signals for churn.

### Recommendations

* **Do Not Deploy**: The model's accuracy is insufficient for production use.
* **Prioritize Data Enrichment**:

  * Customer Sentiment (e.g., NPS surveys)
  * Service Usage Logs (app/portal activity)
  * Marketing Interaction Data
  * **Iterate After Enrichment**:

  * Once enriched features are added, re-run the pipeline to evaluate performance improvement.

---

