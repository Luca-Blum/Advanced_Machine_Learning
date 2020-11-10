# Task 1
## Task description
Given an MRI-scan find the age of the subject. &rightarrow; Regression task
## Problems to overcome
To achieve the best result outlier detection and feature selection has to be perfomed.
## Best parameters

### NaN replacement
- Group all samples (patients) by their label (age) and compute the average for every feature. 

### Isolation Forest
- contamination = 0.04

### KBest
- score_func = f_regression
- k = 210

### SVR
- kernel = 'rbf'
- C = 89