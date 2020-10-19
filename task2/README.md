# Task 2
## Task description
Multi-class classification where you have 3 classes. However, we have changed the original **image** features in several ways. 

Class balance: 600, 3600, 600 **the same split is also in the test set!**
## Problems to overcome
- feature selection
- class imbalance

## Score
```
from sklearn.metrics import balanced_accuracy_score
BMAC = balanced_accuracy_score(y_true, y_pred)
```