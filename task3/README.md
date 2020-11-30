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
``
## Final scripts
- Feature engineering
	- [FeatureEngineering.py]() &rightarrow; "X_train_features.csv"
	- [Yanick.ipynb]() &rightarrow; "X_train_features_yanick.csv"
- Classifier
## Takeaways
- When aggregating data it is important to take many statistical measures, not just the mean
- Just because a model should not be affected by a feature transformation in theory, does not mean that this also happens in practice
- More laborious models don't always result in better scores
- Tuning hyperparameters can give some performance uplift, but more (good) features can have an even bigger impact

## Description

We used [neurokit 2](https://github.com/neuropsychology/NeuroKit) and [Biosppy](https://biosppy.readthedocs.io/en/stable/biosppy.html) to get amplitudes of P,Q,R,S,T peak, length of QRS, a quality score, HRV data (in the frequency domain) and common intervals. We aggregated per patient and took mean, median, standard deviation 5th and 95th percentile.

Because some data was flipped we had to check whether to flip it back. Usually the interval was found by the libraries, but in the other direction, meaning it found the Q spike and meant it was the R spike, so we had to do a little test to potentially revert it back.

The model that we then used started with a [SimpleImputer(mean='median')](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html), [FeatureSelection](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)([RandomForestClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) and then a [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier) of three [HistGradientBoostingClassifiers](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) and two [RandomForestClassifiers](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) each with slightly different parameters and seeds.


Other methods tested were:
- [1DCNN classifier]() on the segment
- [Deep Neural Net classifier]()
- [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) instead of Voting
- Adding a [OneVsOne](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html#sklearn.multiclass.OneVsOneClassifier)/[OneVsRest](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier) wrapper around the classifier
- Testing without the sample segment