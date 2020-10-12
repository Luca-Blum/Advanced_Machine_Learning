import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.svm import SVR

x_train = pd.read_csv("data/X_train.csv",index_col=0,header = 0)
y_train = pd.read_csv("data/y_train.csv",index_col=0,header = 0)
x_test = pd.read_csv("data/X_test.csv",index_col=0,header = 0)


#Drop constant columns
x_train = x_train.loc[:,x_train.apply(pd.Series.nunique) != 1]
x_test = x_test.loc[:,x_test.apply(pd.Series.nunique) != 1]


#Folds for Cross validation
folds = 5
fold = KFold(n_splits = folds)

#Parameter space
dim_nums = np.arange(5,120,5)
#dim_reds = ['pca','tree','lsvc','kbest']
dim_reds = ['kbest']
#replace_nan_methods = {'Simple':['mean','median','most_frequent'],'Iterative':[1,25,50]}
replace_nan_methods = {'Iterative':[1,2,4,8]}
#contaminations = np.logspace(-2,-1,4)
contaminations = [0.02,0.05,0.1]
alphas = np.logspace(-1,1,5)
l1_ratios = np.linspace(0.1,0.9,1)


#R² score for each parameter
r2_score_avg = np.zeros(len(dim_nums) * len(dim_reds) * sum(map(len,replace_nan_methods.values())) * len(contaminations) * len(alphas) * len(l1_ratios))

#score index
pos = 0

#currently best parameter combination
max_score= -np.infty
max_dim = -1
max_dim_red = ''
max_nan_method =''
max_nan_strategy = ''
max_contamination = -1
max_alpha = -1
max_l1_ratio = -1


f = open('data/parameters2.txt', 'w')
f.write('BEST PARAMETERS\n=================\n\n')
f.close()

#Cross validation loop
for dim_index,dim in enumerate(dim_nums):
    for dim_red in dim_reds:
        for replace_nan_method in replace_nan_methods:
            for replace_nan_strategy in replace_nan_methods[replace_nan_method]:
                for contamination in contaminations:
                    for alpha in alphas:
                        for l1_ratio in l1_ratios:

                            print(dim,dim_red,replace_nan_method,replace_nan_strategy,contamination,alpha,l1_ratio)

                            for train_index, test_index in fold.split(x_train):

                                #Split data in training data and validation data
                                x_trainCV = x_train.values[train_index]
                                x_testCV = x_train.values[test_index]
                                y_trainCV = y_train.values[train_index]
                                y_testCV = y_train.values[test_index]


                                #Replacement of NaN's

                                #Default Simple
                                imp = SimpleImputer(missing_values=np.nan, strategy=replace_nan_strategy)

                                if(replace_nan_method == 'Iterative'):
                                    imp = IterativeImputer(n_nearest_features=replace_nan_strategy)

                                x_trainCV = imp.fit_transform(x_trainCV, y_trainCV)
                                x_testCV = imp.transform(x_testCV)


                                #Outlier detection

                                print('before: ', x_trainCV.shape)

                                iso =IsolationForest(contamination=contamination).fit(x_trainCV,y_trainCV)

                                clfTrain = iso.predict(x_trainCV)
                                clfTest = iso.predict(x_testCV)

                                maskTrain = clfTrain != -1
                                maskTest = clfTest != -1

                                x_trainCV = x_trainCV[maskTrain,:]
                                y_trainCV = y_trainCV[maskTrain]

                                x_testCV = x_testCV[maskTest, :]
                                y_testCV = y_testCV[maskTest]

                                print('after outlier detection: ',x_trainCV.shape)


                                #standardization

                                scaler = StandardScaler()

                                x_trainCV = scaler.fit_transform(x_trainCV)
                                x_testCV = scaler.transform(x_testCV)


                                #Feature selection

                                #Default PCA
                                model = PCA(n_components=dim).fit(x_trainCV,y_trainCV)

                                if(dim_red == 'tree'):
                                   clf = ExtraTreesClassifier().fit(x_trainCV, y_trainCV.ravel())
                                   model = SelectFromModel(clf, prefit=True)

                                elif(dim_red == 'lsvc'):

                                    lsvc = LinearSVC(C=0.01, penalty="l1",dual=False).fit(x_trainCV,y_trainCV.ravel())
                                    model = SelectFromModel(lsvc,prefit=True)

                                elif(dim_red == 'kbest'):
                                    model = SelectKBest(f_classif,k=dim).fit(x_trainCV,y_trainCV.ravel())

                                x_trainCV = model.transform(x_trainCV)
                                x_testCV = model.transform(x_testCV)

                                print('after feature selection: ', x_trainCV.shape)


                                #RegressionSVR
                                regr = SVR(C=alpha,max_iter =100000)
                                regr.fit(x_trainCV, y_trainCV.ravel())
                                y_pred = regr.predict(x_testCV)

                                r2_score_avg[pos] += 1./folds * r2_score(y_testCV,y_pred)


                            #update parameters for best model if current model is better
                            if max_score < r2_score_avg[pos]:
                                max_score = r2_score_avg[pos]
                                max_dim = dim
                                max_dim_red = dim_red
                                max_nan_method = replace_nan_method
                                max_nan_strategy = replace_nan_strategy
                                max_contamination = contamination
                                max_alpha = alpha
                                max_l1_ratio = l1_ratio
                                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                                      '!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW BEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                                      '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')
                                print(max_score, max_dim, max_dim_red, max_nan_method, max_nan_strategy, max_contamination, max_alpha, max_l1_ratio)

                                #store history of optimization
                                f = open('data/parameters2.txt', 'a')
                                f.write('R^2 score:\t\t\t\t\t' + str(max_score) + '\n'+
                                        'Dimensions:\t\t\t\t\t' + str(max_dim) + '\n'+
                                        'Reduction method:\t\t\t'+ str(max_dim_red) + '\n'+
                                        'NaN method \t\t\t\t\t' + str(max_nan_method) + '\n' +
                                        'NaN strategy:\t\t\t\t'+ str(max_nan_strategy) + '\n'+
                                        'Outlier contamination:\t\t'+ str(max_contamination) + '\n'+
                                        'Alpha:\t\t\t\t\t\t'+ str(max_alpha) + '\n'+
                                        'L1_ration:\t\t\t\t\t'+ str( max_l1_ratio) + '\n\n')
                                f.close()


                            #show R² score of current model
                            print(r2_score_avg[pos])
                            #proceed to next model
                            pos +=1


#show R² scores of all models
print(r2_score_avg)

#show best model
print(max_score,max_dim,max_dim_red, max_nan_method, max_nan_strategy, max_contamination, max_alpha, max_l1_ratio)


#Apply best model to train data and predict age using test data

#Replace NaN's

imp = SimpleImputer(missing_values=np.nan, strategy=max_nan_strategy)

if(max_nan_method == 'Iterative'):
    imp = IterativeImputer(n_nearest_features=max_nan_strategy)

x_train = imp.fit_transform(x_train, y_train)
x_test = imp.transform(x_test)


#Standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#feature selection
model = PCA(n_components=max_dim).fit(x_train, y_train)

if (max_dim_red == 'tree'):
    clf = ExtraTreesClassifier().fit(x_train, y_train.values.ravel())
    model = SelectFromModel(clf, prefit=True)

elif (max_dim_red == 'lsvc'):

    lsvc = LinearSVC(C=0.01, penalty="l1",dual = False).fit(x_train, y_train.values.ravel())
    model = SelectFromModel(lsvc, prefit=True)

elif (max_dim_red == 'kbest'):
    model = SelectKBest(f_classif, k=max_dim).fit(x_train, y_train.values.ravel())

x_train = model.transform(x_train)
x_test = model.transform(x_test)


#Regression
regr = SVR(C=max_alpha,max_iter =100000)
regr.fit(x_train, y_train.values.ravel())
y_pred = regr.predict(x_test)


#Write predictions
df = pd.DataFrame(y_pred)
df.to_csv('data/y_pred2.csv',header = ['y'], index_label = 'id')

