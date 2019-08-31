# XGBoost_Explore

### Project Objective
The goal of this project is to explore Extreme Gradient Boosting classifiers in R through xgboost as well as 
hyperparameter tuning through mlr. This utilizes the [Adult dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)
from the 1994 US census bureau database and solves for whether a worker receives more than $50K a year. 

### Required Libraries
* data.table 
* ggplot2 
* plyr 
* dplyr 
* h2o 
* xgboost 
* parallel 
* parallelMap 
* mlr 
* caret 
* doParallel 
* pROC 

## Interesting Findings
Below are training and validation results from the xgboost model with some arbitrary hyperparameters. 

Training Parameters and Hyperparameters

| Metric                | Value             |
| :-------------------- | ----------------- |
| booster               | gbtree            |
| objective             | binary:logistic   |
| eta                   | 0.3               |
| gamma                 | 0                 |
| max_depth             | 6                 |
| min_child_weight      | 1                 |
| subsample             | 1                 |
| colsample_bytree      | 1                 |
| nrounds               | 100               |
| nfold                 | 5                 |
| metrics               | auc               |

Cross Validation and Test Results

| Metric                | Value             |
| :-------------------- | ----------------- |
| best iteration        | 50                |
| train_auc_mean (CV)   | 0.9499            |
| test_auc_mean (CV)    | 0.9279            |
| test_roc_auc (Test)   | 0.9276            |

**Test ROC Plot**
 
![XGBoost Test ROC Plot](https://raw.githubusercontent.com/crcastillo/XGBoost_Explore/master/Images/XGBoost%20ROC%20Plot.png)

Below are training and validation results from the xgboost model with a randomized search of hyperparameters (mlr).

Training Parameters and Hyperparameters (Ranges)

| Metric                | Value             |
| :-------------------- | ----------------- |
| booster               | gbtree            |
| objective             | binary:logistic   |
| eta                   | 0.1               |
| gamma                 | 0                 |
| max_depth             | 3:10              |
| min_child_weight      | 1:10              |
| subsample             | 0.5:1             |
| colsample_bytree      | 0.5:1             |
| nrounds               | 100               |
| nfold                 | 5                 |
| metrics               | auc               |

Cross Validation and Test Results

| Metric                | Value             |
| :-------------------- | ----------------- |
| test_auc_mean (CV)    | 0.9288            |
| test_roc_auc (Test)   | 0.9284            |

**Test ROC Plot**

![XGBoost Test ROC Plot Random Search](https://raw.githubusercontent.com/crcastillo/XGBoost_Explore/master/Images/XGBoost%20ROC%20Plot%20-%20Rand.png)


Below are training and validation results from the xgboost model with a grid search of hyperparameters (mlr).

Training Parameters and Hyperparameters (List Values)

| Metric                | Value             |
| :-------------------- | ----------------- |
| booster               | gbtree            |
| objective             | binary:logistic   |
| eta                   | 0.1, 0.2, 0.3     |
| gamma                 | 0, 5, 10, 20      |
| max_depth             | 2, 4, 6, 8, 10    |
| min_child_weight      | 1, 2              |
| subsample             | 1                 |
| colsample_bytree      | 1                 |
| nrounds               | 100               |
| nfold                 | 5                 |
| metrics               | auc               |

Cross Validation and Test Results

| Metric                | Value             |
| :-------------------- | ----------------- |
| test_auc_mean (CV)    | 0.9291            |
| test_roc_auc (Test)   | 0.8942            |

**Test ROC Plot**

![XGBoost Test ROC Plot Grid Search](https://raw.githubusercontent.com/crcastillo/XGBoost_Explore/master/Images/XGBoost%20ROC%20Plot%20-%20Grid.png)