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
## Takeaways
- Some scores and models can be very sensible to small changes of the right models.
- Still, some first decisions and thwarting of models can be made by very coarse estimations, to focus on finer estimations.
- When deciding it is important to have a look at multiple splits and also look at additional information such as confusion matrices.
- Luck is as always beneficial.

## Description
First Kernel PCA (scikit learn) was used to reduce the data set to 400 dimensions (kernel = 'rbf'). The number of dimensions was determined by a rather coarse grained cross-validation. Additionally other kernels were tested but all performed worse. Then ovesampling with the Synthetic Minority Oversampling Technique (SMOTE) using the default values was applied. Afterwards we downsampled again with RandomUnderSample using the default values (both imblearn). Finally we stacked 4 SVC's where the first SVC used a sigmoid, the second a rbf, the third a polynom and the fourth a rbf kernel function. The regularization parameter were determined by cross validation. All other parameters were set to the default values. The best values for the regularization were: 
svc_sig__C: [1.0]
svc_rbf__C: [0.21544346900318834]
svc_poly__C: [0.1]
final_estimator__C: [0.46415888336127786]
Since training the SVC's consumed a lot of time, not that much parameter could be tested. For each SVC 7 different values for the regularization was tested. This resulted in 2401 models and several days of computation. 

