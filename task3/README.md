# Task 3
## Task description
While the previous projects dealt with medical image features, we turn now to the classification of entire time series into one of **4 classes**. This time you will work with the original **ECG recordings** of **different length** sampled as 300Hz to predict heart rhythm. 
## Problems to overcome
- Different lengths of samples
- Time series
## Score function
```
from sklearn.metrics import f1_score
F1 = f1_score(y_true, y_pred, average='micro')
```
## Baselines
Hard: 0.82

Medium: 0.77