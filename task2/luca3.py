import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline

import numpy as np

x_train = pd.read_csv("data/X_train.csv", index_col=0, header=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0, header=0)
x_test = pd.read_csv("data/X_test.csv", index_col=0, header=0)

C_sig = np.logspace(start=-1, stop=1, num=3)
C_rbf = np.logspace(start=-1, stop=1, num=3)
C_poly = np.logspace(start=-1, stop=1, num=3)
C_final = np.logspace(start=-1, stop=1, num=3)
dims = np.linspace(start=100, stop=600, num=6)

max_C_sig = 0
max_C_rbf = 0
max_C_poly = 0
max_C_final = 0
max_dim = 0

max_score = 0

f = open('data/parameters3.txt', 'w')
f.write('BEST PARAMETERS\n'
        '=================\n\n')
f.close()

cv = StratifiedKFold(n_splits=5)

for sig in C_sig:
    for rbf in C_rbf:
        for poly in C_poly:
            for final in C_final:
                for dim in dims:

                    print("Iteration:\nC_sig: ", sig, "\nC_rbf: ", rbf,
                          "\nC_poly: ", poly, "\nC_final: ", final, "\ndim: ", dim, "\n")

                    estimators = [
                        ('svc_sig', SVC(C=sig, kernel='sigmoid', random_state=42)),
                        ('svc_rbf', SVC(C=rbf, kernel='rbf', random_state=42)),
                        ('svc_poly', SVC(C=poly, kernel='poly', random_state=42)),
                        ('gnb',  GaussianNB())
                    ]

                    over = SMOTE(random_state=42)
                    under = RandomUnderSampler(random_state=42)
                    feature_sel = KernelPCA(n_components=dim, kernel='rbf', random_state=42)
                    model = StackingClassifier(estimators=estimators, final_estimator=SVC(C=final, random_state=42))

                    steps = [('feature sel', feature_sel), ('over', over), ('under', under), ('model', model)]
                    pipeline = Pipeline(steps=steps)

                    scores = cross_val_score(pipeline, x_train, y_train.values.ravel(),
                                             scoring='balanced_accuracy', cv=cv, n_jobs=-1)
                    bmac = np.mean(scores)
                    print("BMAC: %.3f" % bmac)

                    if max_score < bmac:

                        max_score = bmac
                        max_C_sig = sig
                        max_C_rbf = rbf
                        max_C_poly = poly
                        max_C_final = final
                        max_dim = dim

                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                              '!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW BEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                              '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')
                        print("score: ", max_score, "\nmax_C_sig: ", max_C_sig, "\nmax_C_rbf: ", max_C_rbf,
                              "\nmax_C_poly: ", max_C_poly, "\nmax_C_final: ", max_C_final,
                              "\nmax_dim: ", max_dim, "\n")

                        # store history of optimization
                        f = open('data/parameters3.txt', 'a')
                        f.write('BMAC score:\t\t\t\t\t' + str(max_score) + '\n' +
                                'C sigmoid:\t\t\t\t\t' + str(max_C_sig) + '\n' +
                                'C rbf:\t\t\t\t\t\t' + str(max_C_rbf) + '\n' +
                                'C poly:\t\t\t\t\t\t' + str(max_C_poly) + '\n' +
                                'C final:\t\t\t\t\t' + str(max_C_final) + '\n' +
                                'dim:\t\t\t\t\t\t' + str(max_dim) + '\n\n')
                        f.close()

print("score: ", max_score, "\nmax_C_sig: ", max_C_sig, "\nmax_C_rbf: ", max_C_rbf,
      "\nmax_C_poly: ", max_C_poly, "\nmax_C_final: ", max_C_final, "\nmax_C_gnb: ", max_dim, "\n")

estimators = [
    ('svc_sig', SVC(C=max_C_sig, kernel='sigmoid', random_state=42)),
    ('svc_rbf', SVC(C=max_C_rbf, kernel='rbf', random_state=42)),
    ('svc_poly', SVC(C=max_C_poly, kernel='poly', random_state=42)),
    ('gnb', GaussianNB())
]

over = SMOTE(random_state=42)
under = RandomUnderSampler(random_state=42)
feature_sel = KernelPCA(dim=max_dim, kernel='rbf', random_state=42)
model = StackingClassifier(estimators=estimators, final_estimator=SVC(C=max_C_final, random_state=42))

steps = [('feature sel', feature_sel), ('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)

pipeline.fit(x_train, y_train.values.ravel())
y_pred = pipeline.predict(x_test)

df = pd.DataFrame(y_pred)
df.to_csv('data/y_pred.csv', header=['y'], index_label='id')
