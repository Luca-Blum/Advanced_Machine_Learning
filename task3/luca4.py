import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif, mutual_info_classif, \
    SelectFromModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

x_train = pd.read_csv("data/X_train_features.csv", index_col=0, header=0, low_memory=False)
y_train = pd.read_csv("data/y_train.csv", index_col=0, header=0)
x_test = pd.read_csv("data/X_test_features.csv", index_col=0, header=0, low_memory=False)

"""
Open/create a file to store the progress of GridSearchCV
and to determine the version number.
The version number ensures that the results get stored in a new file
"""

version = 0
with open('data/version.txt', 'a+') as f:
    f.seek(0)
    read = f.readline().strip()
    if read != '':
        version = int(read) + 1

with open('data/version.txt', 'w') as f:
    print(version, file=f)


# TODO find model above 0.82

# Folds for CV
cv = StratifiedKFold(n_splits=5)

selector = VarianceThreshold()

imputer = SimpleImputer()

scaler = StandardScaler()

# Feature Selection
# feature_sel = SelectKBest(score_func=mutual_info_classif)

feature_sel = SelectFromModel(RandomForestClassifier())


# estimators for Stacking Classifier
estimators = [
    ('mlp1', MLPClassifier(max_iter=1000)),
    ('rfc1', RandomForestClassifier()),
    ('mlp2', MLPClassifier(max_iter=1000))
]

# classification
model = StackingClassifier(estimators=estimators, final_estimator=SVC())


# Parameter space for CV

param_grid = {
    'imputer__strategy': ['median'],

    'classification__mlp1__alpha': [1],

    'classification__rfc1__n_estimators': [150],

    'classification__mlp2__alpha': [1],
    'classification__mlp2__activation': ['tanh'],

    'classification__final_estimator__C': [1],
    'classification__final_estimator__kernel': ['poly']
}

# model to train
steps = [('selector', selector),
         ('imputer', imputer),
         ('scaler', scaler),
         ('feature', feature_sel),
         ('classification', model)]

pipeline = Pipeline(steps=steps)

# CV
search = GridSearchCV(pipeline, param_grid, scoring='f1_micro', cv=cv, n_jobs=-1, verbose=10)
search.fit(x_train, y_train.values.ravel())

# Write results to disk
with open('data/GridSearchCV' + str(version) + '.txt', 'w') as f:

    print("\nBest parameters:", file=f)
    print(search.best_params_, file=f)

    print("\nBest mean score:\t", file=f)
    print(search.best_score_, file=f)

    print("\nSplit scores:", file=f)
    print("split 0: \t\t", search.cv_results_['split0_test_score'][search.best_index_], file=f)
    print("split 1: \t\t", search.cv_results_['split1_test_score'][search.best_index_], file=f)
    print("split 2: \t\t", search.cv_results_['split2_test_score'][search.best_index_], file=f)
    print("split 3: \t\t", search.cv_results_['split3_test_score'][search.best_index_], file=f)
    print("split 4: \t\t", search.cv_results_['split4_test_score'][search.best_index_], file=f)

    print("\nCV history:", file=f)
    print(search.cv_results_, file=f)


# Make predictions on trained model
y_pred = search.predict(x_test)

# store predictions to disk
df = pd.DataFrame(y_pred)
df.to_csv('data/y_pred' + str(version) + '.csv', header=['y'], index_label='id')
