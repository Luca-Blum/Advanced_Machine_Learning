# Task 1
## Task description
Given an MRI-scan find the age of the subject. &rightarrow; Regression task
## Problems to overcome
To achieve the best result outlier detection and feature selection has to be perfomed.
## Best parameters

### Iterative Imputer
- missing_values = np.nan
- n_nearest_feauters = 10

### Isolation Forest
- contamination = 0.04

### KBest
- score_func = f_regression
- k = 92

### SVR
- C = 46.415