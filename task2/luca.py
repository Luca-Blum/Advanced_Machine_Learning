import pandas as pd
import numpy as np

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE

x_train = pd.read_csv("data/X_train.csv", index_col=0, header=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0, header=0)
x_test = pd.read_csv("data/X_test.csv", index_col=0, header=0)

# Folds for Cross validation
folds = 5
fold = KFold(n_splits=folds, shuffle=True, random_state=42)

"""
Strategy:
    - Standardization
    - Feature reduction => KBest
    - Up/Down sampling => SMOTE, BorderlineSMOTE, SVMSMOTE
        k_neighbors
    - Classification => SVC
"""

# Parameter space
dims = np.arange(start=50, stop=400, step=20)
smotes = ['SMOTE' ]#, 'BorderlineSMOTE', 'SVMSMOTE', 'None']
k_neighbors = np.arange(start=1, stop=11, step=2)
Cs = np.logspace(-1, 2, 10)

# R² score for each parameter
BMACs = np.zeros(len(dims) * len(smotes) * len(k_neighbors) * len(Cs))

# score index
pos = 0

# currently best parameter combination
max_score = -np.infty
max_dim = -1
max_smote = ''
max_k_neighbor = -1
max_c = -1

f = open('data/parameters.txt', 'w')
f.write('BEST PARAMETERS\n'
        '=================\n\n')
f.close()

# Cross validation loop
for dim in dims:
    for smote in smotes:
        print(dim, smote)
        for k_neighbor in k_neighbors:
            for c in Cs:

                for train_index, test_index in fold.split(x_train):

                    # Split data in training data and validation data
                    x_trainCV = x_train.values[train_index]
                    x_testCV = x_train.values[test_index]
                    y_trainCV = y_train.values[train_index]
                    y_testCV = y_train.values[test_index]

                    # Standardization
                    scalerCV = StandardScaler()

                    x_trainCV = scalerCV.fit_transform(x_trainCV)
                    x_testCV = scalerCV.transform(x_testCV)

                    # Feature Selection
                    feature_selCV = SelectKBest(f_classif, k=dim).fit(x_trainCV, y_trainCV.ravel())

                    x_trainCV = feature_selCV.transform(x_trainCV)
                    x_testCV = feature_selCV.transform(x_testCV)

                    # Upsampling with Synthetic Minority Over-sampling (SMOTE)
                    smCV = SMOTE(random_state=42, k_neighbors=k_neighbor, n_jobs=-1)

                    if smote == 'BorderlineSMOTE':
                        smCV = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbor, n_jobs=-1)
                    elif smote == 'SVMSMOTE':
                        smCV = SVMSMOTE(random_state=42, k_neighbors=k_neighbor, n_jobs=-1)

                    if smote != 'None':
                        x_trainCV, y_trainCV = smCV.fit_resample(x_trainCV, y_trainCV)

                    # Classification
                    classifierCV = SVC(C=c, random_state=42)
                    classifierCV.fit(x_trainCV, y_trainCV.ravel())
                    y_predCV = classifierCV.predict(x_testCV)

                    BMACs[pos] += 1./folds * balanced_accuracy_score(y_testCV, y_predCV)

                # update parameters for best model if current model is better
                if max_score < BMACs[pos]:
                    max_score = BMACs[pos]
                    max_dim = dim
                    max_smote = smote
                    max_k_neighbor = k_neighbor
                    max_c = c

                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                          '!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW BEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                          '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')
                    print("score: ", max_score, "\ndim: ", max_dim,
                          "\nsmote: ", max_smote, "\nk_neighbors: ", max_k_neighbor,
                          "\nC: ", max_c, "\n")

                    # store history of optimization
                    f = open('data/parameters.txt', 'a')
                    f.write('BMAC score:\t\t\t\t\t' + str(max_score) + '\n' +
                            'Dimensions:\t\t\t\t\t' + str(max_dim) + '\n' +
                            'SMOTE:\t\t\t\t\t\t' + str(max_smote) + '\n' +
                            'k_neighbor:\t\t\t\t\t' + str(max_k_neighbor) + '\n' +
                            'C:\t\t\t\t\t\t\t' + str(max_c) + '\n\n')
                    f.close()

                # show BMAC score of current model
                print(BMACs[pos])
                # proceed to next model
                pos += 1

# show BMAC scores of all models
print(BMACs)

# show best model
print("score: ", max_score, "\ndim: ", max_dim,
      "\nsmote: ", max_smote, "\nk_neighbors: ", max_k_neighbor,
      "\nC: ", max_c, "\n")

# Apply best model to train data and predict age using test data

# Standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Feature selection
feature_sel = SelectKBest(f_classif, k=max_dim).fit(x_train, y_train.values.ravel())

x_train = feature_sel.transform(x_train)
x_test = feature_sel.transform(x_test)

# Upsampling
sm = SMOTE(random_state=42, k_neighbors=max_k_neighbor)

if max_smote == 'BorderlineSMOTE':
    sm = BorderlineSMOTE(random_state=42, k_neighbors=max_k_neighbor)
elif max_smote == 'SVMSMOTE':
    sm = SVMSMOTE(random_state=42, k_neighbors=max_k_neighbor)

x_train, y_train = sm.fit_resample(x_train, y_train)

# Classification
classifier = SVC(C=max_c, random_state=42)
classifier.fit(x_train, y_train.values.ravel())
y_pred = classifier.predict(x_test)

df = pd.DataFrame(y_pred)
df.to_csv('data/y_pred.csv', header=['y'], index_label='id')
