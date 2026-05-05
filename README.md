# STAT 5243 Project 4: U.S. Flight Delay Analysis and Prediction

## Project Overview

This project is an end-to-end machine learning project for STAT 5243. We analyze U.S. domestic flight on-time performance data from January to March 2025. The main goal is to understand flight delay patterns and build supervised machine learning models to predict whether a flight is likely to experience an arrival delay.

The project follows a full data science workflow, including data collection, data cleaning, exploratory data analysis, unsupervised learning, feature engineering, preprocessing, supervised modeling, model comparison, and interpretation.

## Research Question

Can flight schedule information, carrier information, airport information, route information, and other operational features be used to predict whether a flight will be delayed?

## Data Source

The raw data comes from the Bureau of Transportation Statistics (BTS) On-Time Performance dataset.

The original monthly raw data files used in this project are stored in the root directory:

- `T_ONTIME_MARKETING2025_01.csv`
- `T_ONTIME_MARKETING2025_02.csv`
- `T_ONTIME_MARKETING2025_03.csv`

These files correspond to U.S. flight records from January, February, and March 2025.

The combined raw dataset and cleaned dataset are stored in the `outputs/` folder:

- `outputs/flight_delay_raw_2025_Q1.csv.zip`
- `outputs/flight_delay_clean_2025_Q1.csv.zip`

`flight_delay_raw_2025_Q1.csv.zip` contains the combined raw dataset for 2025 Q1.

`flight_delay_clean_2025_Q1.csv.zip` contains the cleaned dataset generated after the data collection and cleaning step.

## Large Files and Google Drive Link

Some large log files and supplementary output files are not uploaded directly to GitHub because of GitHub file size limits. These files are available in the following Google Drive folder:

[Google Drive Folder](https://drive.google.com/drive/folders/1vq-45joV25rt7CYExV2YWiJrdO1vb3oJ?usp=drive_link)

Please make sure to download these files if you want to reproduce the full workflow with all intermediate logs and outputs.

The main code files, raw data files, cleaned datasets, processed modeling files, and key results are included in this GitHub repository.

## Repository Structure

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
