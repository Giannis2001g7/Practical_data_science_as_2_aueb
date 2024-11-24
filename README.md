Master in Data Science AUEB 2024-2025

Practical Data Science Assignment 2

Ioannis Papadopoulos f3352409

FOOD HAZARD DETECTION CHALLENGE

Introduction This repository contains a Python-based machine learning pipeline designed to classify hazards and products into predefined categories. The solution supports both basic (Logistic Regression) and advanced (XGBOOST) machine learning models, offering
flexibility and modularity. The code is optimized for use in environments like Google Colab and processes textual data for training, testing, and prediction.

Requirements

The code runs in Google Colab and requires the following Python libraries:

pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

re

zipfile

Ensure all necessary packages are installed in your Colab environment before executing the code.

Dataset

The project expects two CSV files:

incidents_train.csv - Training dataset with features title, text, and target labels hazard-category and product-category.

incidents.csv - Testing dataset with title and text columns for predictions.

These files should be uploaded manually in the Colab environment when prompted.

Key Features

1. Data Cleaning

Text cleaning functions remove special characters, convert text to lowercase, and strip whitespace.

title and text columns are preprocessed for both training and test datasets.

2. Exploratory Data Analysis

Visualizes class distributions for hazard-category and product-category using bar plots.

3. Model Training (ST1 and ST2 Tasks)

ST1 Task:

Predicts hazard-category and product-category separately.

Utilizes Logistic Regression and XGBoost models.

TF-IDF vectorization is applied to the title and text columns.

ST2 Task:

Combines the prediction of hazard and product categories.

Handles rare classes by merging them into an "other" category.

Uses the same models (Logistic Regression and XGBoost) and TF-IDF vectorization.

4. Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

These metrics are computed for all models and tasks during validation.

5. Predictions and Output

Predictions for each model and feature are saved as CSV files.

Outputs are compressed into ZIP files (predictions_ST1.zip and predictions_ST2.zip).

6. Submission File

Predictions from ST1 and ST2 tasks are merged and saved as submission.csv.

The merged file is compressed into submission.zip for final submission.

How to Use

Upload the required datasets when prompted.

Run the code cells sequentially in Google Colab.

Observe printed metrics and visualizations.

Download the resulting ZIP files (predictions_ST1.zip, predictions_ST2.zip, and submission.zip).

Key Functions

Data Preprocessing

clean_text(text): Cleans input text by removing special characters, converting to lowercase, and trimming whitespace.

Training and Evaluation

train_model_st1: Trains models for ST1 task with TF-IDF vectorization.

evaluate_model: Computes evaluation metrics for model predictions.

handle_rare_classes: Merges rare classes into an "other" category.

train_model_st2: Trains models for ST2 task with label encoding and TF-IDF vectorization.

Predictions

predict_test_data: Predicts test data labels for ST1 task.

predict_test_data_st2: Predicts test data labels for ST2 task.

Utilities

compress_csv_files: Compresses CSV files into a ZIP archive.

Output Structure

predictions_ST1/: Contains predictions for ST1 task.

predictions_ST2/: Contains predictions for ST2 task.

submission.csv: Final merged predictions.

submission.zip: Compressed file for final submission.

Notes

Adjust the threshold in handle_rare_classes to customize the merging of rare classes.

Ensure that the datasets contain all required columns as per the code specifications.

Future Enhancements

Expand the pipeline to support additional models and hyperparameter tuning.

Add more sophisticated handling for class imbalance.

Improve data visualization to identify trends and correlations.

Results 
ST1: `hazard-category` and `product-category` - Score: 0.4466 - Ranking: 33th place 
ST2: `hazard` and `product` - Score: 0.1811 - Ranking: 23th place
