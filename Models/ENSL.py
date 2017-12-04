from __future__ import division

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer

import random
import numpy as np
import copy
import csv as csv




#################################Ensemble Learning Model############################################



#return prediction r_np over X_test
def stackingClassification(model1,model2,X_train,Y_train,X_test,Y_test):
    #############merge X with Y###########################################################################
    X_train = X_train.toarray();
    X_test =  X_test.toarray();
    estimators=[]
    estimators.append(model1);
    estimators.append(model2);
    train = copy.deepcopy(X_train)
    for i in range(0,len(X_train)):
        train[i].append(Y_train[i]);
        test = copy.deepcopy(X_test)

    for i in range(0,len(X_test)):
        test[i].append(Y_test[i]);

    print "Block"

    ##########create five fold with a new column foldID#################################################
    random.shuffle(train);
    for i in range(0,len(train)):
        train[i].insert(1,i%5 + 1);
    
    random.shuffle(train);

    ##########Create train_meta and test_meta(copy of training and test set, with new 2 empty columns)##
    train_meta = copy.deepcopy(train);
    test_meta= copy.deepcopy(test);
    for i in range(0,len(train)):
        train_meta[i].insert((len(train[i])-1),np.nan);
        train_meta[i].insert((len(train[i])-1),np.nan);
    for i in range(0,len(test)):
        test_meta[i].insert((len(test[i])-1),np.nan)
        test_meta[i].insert((len(test[i])-1),np.nan)


    ##########train on fold2 to fold 5 and fit on fold 1################################################
    ######################separate fold####################
    fold_array=[];
    for i in range(0,5):
        fold_array.append([]);

    for i in range (0, len(train_meta)):
        fold_array[train_meta[i][1]-1].append(train_meta[i]);


    #########Put training data of each fold as a dataset without the two columns########################
    X_train_fold=[];
    Y_train_fold=[];

    for i in range (0,5):
        X_train_fold.append([]);
        Y_train_fold.append([]);


    for i in range (0,5):
        for k in range (0, len(fold_array[i])):
            X_train_fold[i].append([]);
            for j in range (0, len(fold_array[i][k]) -3):
                if j != 1 :
                    X_train_fold[i][k].append(fold_array[i][k][j]);
                    Y_train_fold[i].append(fold_array[i][k][len(fold_array[i][k])-1]);
                


    first_X_train =[]
    first_Y_train =[]
    for i in range (1,5):
        for j in range (0, len(X_train_fold[i])):
            first_X_train.append(X_train_fold[i][j]);
            first_Y_train.append(Y_train_fold[i][j]);


    classifier=[]
    for i in range (0, len(estimators)):
        classifier.append(estimators[i].fit(first_X_train, first_Y_train));

    r=[];
    for j in range (0,len(classifier)):
        r.append(predict(X_train_fold[0],classifier[j]));
        r[j] =  list(r[j])

    for i in range (0, len(fold_array[0])):
        fold_array[0][i][len(fold_array[0][i])-3] = r[0][i];
        fold_array[0][i][len(fold_array[0][i])-2] = r[1][i];


    train_meta=[]
    for i in range (0,len(fold_array)):
        for j in range(0,len(fold_array[i])):
            train_meta.append(fold_array[i][j]);
    r=[];
    for i in range (0, 2):
        classifier = estimators[i].fit(X_train,Y_train);
        r.append( predict(X_test,classifier));

    for i in range (0,len(test_meta)):
        test_meta[i][len(test_meta[i])-3] = r[0][i];
        test_meta[i][len(test_meta[i])-2] = r[1][i];


    #######################Stacking model (logistic reression) ###########################
    stacking_model = LogisticRegression()

    X_train_meta=[]
    Y_train_meta=[]
    for i in range (0, len(train_meta)):
        X_train_meta.append([]);
        for j in range (0, len(train_meta[i])-1):
            if (j!=1) :
                X_train_meta[i].append(train_meta[i][j]);
                Y_train_meta.append(train_meta[i][len(train_meta[i])-1]);


    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_train_meta = imp.fit_transform(X_train_meta);


    stacking_clf= stacking_model.fit(X_train_meta,Y_train_meta);

    X_test_meta=[]
    for i in range (0,len(test_meta)):
        X_test_meta.append([]);
        for j in range (0, len(test_meta[i])-1):
            X_test_meta[i].append(test_meta[i][j]);
        
    r_np= predict(X_test_meta,stacking_clf);

    return r_np
