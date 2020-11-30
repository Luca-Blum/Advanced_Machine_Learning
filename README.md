# Advanced Machine Learning

## Purpose
This repository is used for the development of the projects for the Advanced Machine Learning course of  Prof. Joachim M. Buhmann lectured at ETH Zürich during the autumn semester 2020.

## Task 0
Dummy task to get familiar with the project system and to develop a basic template for frequently used functionalities like input/output, splitting data in train and test set and building a simple neural network.

## [Task 1](https://github.com/lblum95/AML/blob/master/task1/README.md)
This task has problems concerning:
- **Incomplete data** &rightarrow; Grouping and averaging
- **Outlier detection** &rightarrow; [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- **Feature selection** &rightarrow; [KBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)

We managed to solve these problems good enough to pass the hardest baseline.
## [Task 2](https://github.com/lblum95/AML/blob/master/task2/README.md)
This task has problems concerning:
- **Imbalanced data** &rightarrow; [SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html)
- **Sensitive score** &rightarrow; [Stratified KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

We managed to solve these problems exceptionally well, such that we placed third on that homework. &#129395;
## [Task 3](https://github.com/lblum95/AML/blob/master/task3/README.md)
This task has problems concerning:
- **Time series** &rightarrow; [Transform to frequency domain with HRV](https://neurokit2.readthedocs.io/en/latest/examples/hrv.html#Compute-HRV-features)
- **Different lengths of samples** &rightarrow; Take aggregates e.g. mean, 5th/95th percentage
- **Challenging classification** &rightarrow; [HistGradientClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
## Task 4
