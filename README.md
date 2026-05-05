# STAT 5243 Project 4: U.S. Flight Delay Analysis and Prediction

This project also includes an optional Shiny web application:

- `app.R`

The app provides an interactive way to view selected project results and supports the optional interactive product component of the project.

The app can be accessed here:

[Shiny App Link](https://zhuyunj.shinyapps.io/5243project4/)

## 1. Project Overview

This project is an end-to-end machine learning project for STAT 5243. We analyze U.S. domestic flight on-time performance data from January to March 2025. The main goal of this project is to understand patterns in flight delays and build supervised machine learning models to predict whether a flight is likely to experience an arrival delay.

The project follows a complete workflow:

1. Data collection and cleaning
2. Exploratory data analysis
3. Unsupervised learning
4. Feature engineering and preprocessing
5. Supervised model development
6. Model comparison and evaluation
7. Interpretation of results

## 2. Research Question

Can flight schedule information, carrier information, airport information, route information, and other operational features be used to predict whether a flight will be delayed?

## 3. Data Source

The data used in this project comes from the Bureau of Transportation Statistics (BTS) On-Time Performance dataset.

The original monthly raw data files are stored in the root directory of this repository:

* `T_ONTIME_MARKETING2025_01.csv`
* `T_ONTIME_MARKETING2025_02.csv`
* `T_ONTIME_MARKETING2025_03.csv`

These files correspond to U.S. flight records from January, February, and March 2025.

The combined and cleaned datasets are stored in the `outputs/` folder:

* `outputs/flight_delay_raw_2025_Q1.csv.zip`
* `outputs/flight_delay_clean_2025_Q1.csv.zip`

`flight_delay_raw_2025_Q1.csv.zip` contains the combined raw dataset for January to March 2025.

`flight_delay_clean_2025_Q1.csv.zip` contains the cleaned dataset generated after the data collection and cleaning step.

## 4. Large Files and Google Drive Link

Some large log files and supplementary output files are not uploaded directly to GitHub because of GitHub file size limits. These files are available in the following Google Drive folder:

[Google Drive Folder](https://drive.google.com/drive/folders/1vq-45joV25rt7CYExV2YWiJrdO1vb3oJ?usp=drive_link)

Please download these files if you want to reproduce the full workflow with all intermediate logs and outputs. Use columbia.edu account to login.

The main code files, raw data files, cleaned datasets, processed modeling files, and key results are included in this GitHub repository.

## 5. Repository Structure

```text
5243project4/
│
├── 5243p4.ipynb
├── eda_unsup.ipynb
├── feature_engineering + preprocessing lead.py
├── model_development_Zhuyun.ipynb
│
├── T_ONTIME_MARKETING2025_01.csv
├── T_ONTIME_MARKETING2025_02.csv
├── T_ONTIME_MARKETING2025_03.csv
│
├── X_train_processed.npz
├── X_test_processed.npz
├── y_train.csv
├── y_test.csv
├── processed_feature_names.csv
├── preprocessing_pipeline.joblib
├── feature_summary.json
├── 5243 project4.pptx
├── app.R
└── rsconnect/
    └── shinyapps.io/
        └── zhuyunj/
            └── 5243project4.dcf
│
└── outputs/
    ├── airportid1.csv
    ├── airportid2.csv
    ├── airportid3.csv
    ├── flight_delay_raw_2025_Q1.csv.zip
    ├── flight_delay_clean_2025_Q1.csv.zip
    ├── debug_model_results.csv
    ├── model_cv_results.csv
    ├── final_model_comparison_table.csv
    ├── final_test_results.csv
    ├── final_xgboost_model.joblib
    ├── xgboost_best_params.csv
    ├── xgboost_feature_importance.csv
    ├── xgboost_feature_importance_named.csv
    ├── xgboost_threshold_tradeoff.csv
    ├── confusion_matrix_xgboost.png
    ├── roc_curve_xgboost.png
    ├── precision_recall_curve_xgboost.png
    └── additional model evaluation figures and tables
```

## 6. File Descriptions

### 6.1 `5243p4.ipynb`

This notebook performs data collection and cleaning.

It loads the three original monthly raw data files:

* `T_ONTIME_MARKETING2025_01.csv`
* `T_ONTIME_MARKETING2025_02.csv`
* `T_ONTIME_MARKETING2025_03.csv`

It combines them into a Q1 2025 dataset, handles missing values, formats variables, and creates the cleaned dataset.

Main outputs:

* `outputs/flight_delay_raw_2025_Q1.csv.zip`
* `outputs/flight_delay_clean_2025_Q1.csv.zip`

### 6.2 `eda_unsup.ipynb`

This notebook performs exploratory data analysis and unsupervised learning.

It uses the cleaned dataset:

* `outputs/flight_delay_clean_2025_Q1.csv.zip`

Main analysis includes:

* Overall flight delay distribution
* Delay patterns by departure time
* Delay patterns by carrier
* Delay patterns by origin airport
* Airport-level delay profiles
* PCA for dimensionality reduction
* K-Means clustering for unsupervised airport-level analysis

Related output files:

* `outputs/airportid1.csv`
* `outputs/airportid2.csv`
* `outputs/airportid3.csv`

### 6.3 `feature_engineering + preprocessing lead.py`

This script performs feature engineering and preprocessing for supervised machine learning.

It prepares the train-test split, transforms variables, encodes categorical variables, and saves the processed feature matrices.

Main outputs:

* `X_train_processed.npz`
* `X_test_processed.npz`
* `y_train.csv`
* `y_test.csv`
* `processed_feature_names.csv`
* `preprocessing_pipeline.joblib`
* `feature_summary.json`

### 6.4 `model_development_Zhuyun.ipynb`

This notebook performs supervised model development and model evaluation.

It uses the processed training and testing files:

* `X_train_processed.npz`
* `X_test_processed.npz`
* `y_train.csv`
* `y_test.csv`
* `processed_feature_names.csv`

Main outputs:

* `outputs/model_cv_results.csv`
* `outputs/final_model_comparison_table.csv`
* `outputs/final_test_results.csv`
* `outputs/final_xgboost_model.joblib`
* `outputs/xgboost_best_params.csv`
* `outputs/xgboost_feature_importance.csv`
* `outputs/xgboost_feature_importance_named.csv`
* `outputs/xgboost_threshold_tradeoff.csv`
* `outputs/confusion_matrix_xgboost.png`
* `outputs/roc_curve_xgboost.png`
* `outputs/precision_recall_curve_xgboost.png`

### 6.5 `5243 project4.pptx`

This file contains the final presentation slides for the project. The slides summarize the project goal, dataset, data cleaning process, exploratory analysis, unsupervised learning insights, feature engineering, supervised modeling results, final model evaluation, and main conclusions. This file is used for the oral presentation component of the project.


## 7. Required Packages

This project mainly uses Python.

Main Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib scipy
```

Jupyter Notebook is also needed if running the notebooks locally:

```bash
pip install notebook jupyter
```

A full installation command is:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib scipy notebook jupyter
```

## 8. How to Run the Code

### Step 1: Clone the repository

```bash
git clone https://github.com/qf2188/5243project4.git
cd 5243project4
```

### Step 2: Install required packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib scipy notebook jupyter
```

### Step 3: Open Jupyter Notebook

```bash
jupyter notebook
```

This will open the project folder in a browser.

### Step 4: Run the data collection and cleaning notebook

Open and run:

```text
5243p4.ipynb
```

This notebook uses the three monthly raw data files in the root directory:

```text
T_ONTIME_MARKETING2025_01.csv
T_ONTIME_MARKETING2025_02.csv
T_ONTIME_MARKETING2025_03.csv
```

Expected outputs:

```text
outputs/flight_delay_raw_2025_Q1.csv.zip
outputs/flight_delay_clean_2025_Q1.csv.zip
```

### Step 5: Unzip the cleaned dataset if needed

If the later notebooks cannot directly read the zipped file, unzip it manually:

```bash
unzip outputs/flight_delay_clean_2025_Q1.csv.zip -d outputs/
```

This should create the cleaned CSV file inside the `outputs/` folder.

### Step 6: Run the EDA and unsupervised learning notebook

Open and run:

```text
eda_unsup.ipynb
```

This notebook uses the cleaned dataset from the `outputs/` folder.

Expected input:

```text
outputs/flight_delay_clean_2025_Q1.csv
```

or, depending on the notebook path:

```text
outputs/flight_delay_clean_2025_Q1.csv.zip
```

Expected outputs include EDA summaries, visualizations, and airport-level files:

```text
outputs/airportid1.csv
outputs/airportid2.csv
outputs/airportid3.csv
```

### Step 7: Run the feature engineering and preprocessing script

Run the Python script:

```bash
python "feature_engineering + preprocessing lead.py"
```

Expected outputs:

```text
X_train_processed.npz
X_test_processed.npz
y_train.csv
y_test.csv
processed_feature_names.csv
preprocessing_pipeline.joblib
feature_summary.json
```

These files are used as inputs for supervised modeling.

### Step 8: Run the model development notebook

Open and run:

```text
model_development_Zhuyun.ipynb
```

This notebook uses the processed modeling files generated in Step 7.

Expected inputs:

```text
X_train_processed.npz
X_test_processed.npz
y_train.csv
y_test.csv
processed_feature_names.csv
```

Expected outputs:

```text
outputs/model_cv_results.csv
outputs/final_model_comparison_table.csv
outputs/final_test_results.csv
outputs/final_xgboost_model.joblib
outputs/xgboost_best_params.csv
outputs/xgboost_feature_importance.csv
outputs/xgboost_feature_importance_named.csv
outputs/xgboost_threshold_tradeoff.csv
outputs/confusion_matrix_xgboost.png
outputs/roc_curve_xgboost.png
outputs/precision_recall_curve_xgboost.png
```

## 9. Running Order Summary

The complete running order is:

```text
1. 5243p4.ipynb
2. eda_unsup.ipynb
3. feature_engineering + preprocessing lead.py
4. model_development_Zhuyun.ipynb
```

The workflow is:

```text
Raw monthly CSV files
        ↓
5243p4.ipynb
        ↓
Combined raw Q1 dataset and cleaned Q1 dataset
        ↓
eda_unsup.ipynb
        ↓
EDA results and unsupervised learning outputs
        ↓
feature_engineering + preprocessing lead.py
        ↓
Processed train/test data
        ↓
model_development_Zhuyun.ipynb
        ↓
Final model comparison, evaluation results, and plots
```

## 10. Methods

This project uses the following methods:

* Data cleaning and missing value handling
* Exploratory data analysis
* Data visualization
* Airport-level aggregation
* PCA for dimensionality reduction
* K-Means clustering for unsupervised airport-level analysis
* Feature engineering
* Categorical variable encoding
* Train-test split
* Supervised classification models
* Cross-validation
* Model comparison
* XGBoost model tuning and evaluation
* Feature importance analysis
* ROC curve analysis
* Precision-recall curve analysis
* Threshold tradeoff analysis

## 12. Output Location

Most output files are stored in the `outputs/` folder.

Important output files include:

```text
outputs/flight_delay_raw_2025_Q1.csv.zip
outputs/flight_delay_clean_2025_Q1.csv.zip
outputs/model_cv_results.csv
outputs/final_model_comparison_table.csv
outputs/final_test_results.csv
outputs/final_xgboost_model.joblib
outputs/xgboost_best_params.csv
outputs/xgboost_feature_importance.csv
outputs/xgboost_feature_importance_named.csv
outputs/xgboost_threshold_tradeoff.csv
outputs/confusion_matrix_xgboost.png
outputs/roc_curve_xgboost.png
outputs/precision_recall_curve_xgboost.png
```

Processed modeling files are stored in the root directory:

```text
X_train_processed.npz
X_test_processed.npz
y_train.csv
y_test.csv
processed_feature_names.csv
preprocessing_pipeline.joblib
feature_summary.json
```

## 13. Team Contributions

Each team member contributed to a different part of the project:

* Data Collection and Cleaning: Qixiang Fan
* Exploratory Data Analysis and Unsupervised Learning: Qingyue Wang
* Feature Engineering and Preprocessing:Yiyu Chen
* Supervised modeling, hyperparameter tuning, and model evaluation: Zhuyun Jin
* Model selection, final report integration: Omari Motta
* Final Report and Presentation: All team members

Please refer to the final report for a more detailed explanation of each member's contribution.

## 14. Notes

Because some datasets and log files are large, they are stored as compressed `.zip` files or provided through the Google Drive folder. Please unzip the data files before running notebooks that require the full combined or cleaned dataset.

