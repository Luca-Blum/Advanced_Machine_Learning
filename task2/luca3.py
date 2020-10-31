import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.decomposition import KernelPCA
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline

import numpy as np

"""
Euler: bsub -n 24 -W 72:00 -N -J "task-2" -oo data/output.txt python3 luca3.py
"""

# Read in data for training and testing
x_train = pd.read_csv("data/X_train.csv", index_col=0, header=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0, header=0)
x_test = pd.read_csv("data/X_test.csv", index_col=0, header=0)

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

# Folds for CV
cv = StratifiedKFold(n_splits=5)

# estimators for Stacking Classifier
estimators = [
    ('svc_sig', SVC(kernel='sigmoid', random_state=42)),
    ('svc_rbf', SVC(kernel='rbf', random_state=42)),
    ('svc_poly', SVC(kernel='poly', random_state=42))
]

# Oversampling
over = SMOTE(random_state=42)
# Undersampling
under = RandomUnderSampler(random_state=42)
# Feature Selection
feature_sel = KernelPCA(kernel='rbf', random_state=42)
# classification
model = StackingClassifier(estimators=estimators, final_estimator=SVC(random_state=42))

# Parameter space for CV
param_grid = {
    'feature__n_components': np.linspace(start=100, stop=600, num=2),
    'classification__svc_sig__C': [1],
    'classification__svc_rbf__C': [1],
    'classification__svc_poly__C': [1],
    'classification__final_estimator__C': [1]
}

# model to train
steps = [('feature', feature_sel), ('over', over), ('under', under), ('classification', model)]
pipeline = Pipeline(steps=steps)

# CV
search = GridSearchCV(pipeline, param_grid, scoring='balanced_accuracy', cv=cv, n_jobs=-1, verbose=10)
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
