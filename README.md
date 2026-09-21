# CreditScorePrediction
End-to-end ML pipeline for credit score classification with model optimization, MLflow tracking, Streamlit deployment, and AWS SageMaker infrastructure.

## Overview

This project develops an end-to-end machine learning system for predicting customer credit scores based on financial and behavioral characteristics.

The project classifies customers into three credit-score categories:

- Poor
- Standard
- Good

The workflow covers the complete machine learning lifecycle, from exploratory data analysis and preprocessing to model experimentation, optimization, experiment tracking, local deployment, and cloud deployment on AWS.

The final system integrates **LightGBM, Optuna, SMOTE, MLflow, Streamlit, Amazon SageMaker, Amazon S3, EC2, and CloudWatch**.

## 🚀 Project Highlights

- **25,000 customer records** analyzed for credit-score classification
- **26 original features** processed and transformed into **49 model-ready features**
- Compared **3 machine learning models**:
  - XGBoost
  - Random Forest
  - LightGBM
- Applied **SMOTE** to address class imbalance
- Used **5-fold cross-validation** for model evaluation
- Performed **50 Optuna optimization trials** with early stopping
- Achieved a best cross-validation **Weighted F1-score of 0.7224**
- Achieved **73% test accuracy** and **0.73 weighted F1-score**
- Implemented **MLflow experiment tracking**
- Built a **Streamlit web application** for model inference
- Deployed the ML pipeline to **AWS SageMaker**
- Used **Amazon S3** for model artifact storage
- Used **EC2** for cloud-based execution
- Used **CloudWatch** for monitoring and troubleshooting

## 🧹 Data Preprocessing

The dataset initially contained inconsistent data types, missing values, invalid numerical values, and noisy categorical values.

The preprocessing process included:

- Removing non-predictive identifier columns
- Converting incorrectly formatted numerical columns
- Removing unwanted characters such as `_`
- Handling invalid and unrealistic numerical values
- Treating negative or out-of-range values as missing
- Handling missing categorical values
- Applying median imputation to numerical variables
- Removing duplicate records
- Encoding categorical variables
- Applying RobustScaler to numerical features
- Applying OrdinalEncoder to Payment Behaviour
- Applying OneHotEncoder to categorical features

After preprocessing, the original **26 input features were transformed into 49 model-ready features**.

## ⚙️ Feature Engineering

Several features were engineered to capture customer financial behavior:

- `Credit_History_Age`
  - Converted from years/months format into total months

- `Savings_Rate`
  - Calculated from monthly balance and monthly income

- `Delay_Ratio`
  - Derived from delayed payments relative to bank accounts and credit cards

- `Credit_Mix_Encoded`
  - Encoded credit mix categories into numerical values

- `Payment_Min_Encoded`
  - Encoded minimum payment behavior

- `Credit_Score_Encoded`
  - Encoded the target credit-score categories
 
## 🤖 Model Experimentation

Three machine learning algorithms were evaluated:

| Model | Approach |
|---|---|
| XGBoost | Gradient boosting classification |
| Random Forest | Ensemble tree-based classification |
| LightGBM | Gradient boosting with leaf-wise tree growth |

To address class imbalance, **SMOTE** was integrated into the training pipeline.

Model selection was performed using **5-fold cross-validation with weighted F1-score** as the optimization objective.

## 🔬 Hyperparameter Optimization

Optuna was used to automatically search for optimal model configurations.

The optimization process:

- Evaluated up to **50 trials**
- Compared XGBoost, Random Forest, and LightGBM
- Optimized model-specific hyperparameters
- Optimized SMOTE `k_neighbors`
- Used **5-fold cross-validation**
- Used **weighted F1-score** as the objective
- Implemented an early-stopping mechanism after **15 consecutive trials without improvement**

The best configuration achieved a cross-validation weighted F1-score of:

**0.7224**

The selected model was:

**LightGBM**

## 📊 Final Model Performance

The optimized LightGBM model was retrained using the training data and evaluated on a 5,000-record test set.

| Metric | Score |
|---|---:|
| Accuracy | 0.73 |
| Weighted Precision | 0.73 |
| Weighted Recall | 0.73 |
| Weighted F1-score | 0.73 |

### Class-Level Performance

| Credit Score | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Poor | 0.73 | 0.74 | 0.73 |
| Standard | 0.76 | 0.77 | 0.76 |
| Good | 0.65 | 0.61 | 0.63 |

The test set contained **5,000 observations**.

### ☁️ AWS Deployment

The deployment pipeline was migrated from a local environment to AWS to support a more scalable architecture.

AWS services used:

- **Amazon SageMaker** — model training/deployment infrastructure
- **Amazon S3** — model artifact storage
- **Amazon EC2** — cloud-based execution environment
- **Amazon CloudWatch** — monitoring and troubleshooting

The trained model artifact was packaged as:

`model.tar.gz`

and uploaded to an Amazon S3 bucket before deployment through SageMaker.
