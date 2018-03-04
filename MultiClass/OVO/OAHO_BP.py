# -*- coding: UTF-8 -*-
from sklearn.neural_network import MLPClassifier
import numpy as np
from decomposeOAHO import decomposeOAHO
from sklearn import metrics
from statAUC import statAUC
from sklearn.model_selection import KFold

# @artical:OAHO-an Effective Algorithm for Multi-Class Learning from Imbalanced Data
# @param trainSet:
# @param testSet:
# @param n_class: class num
# @param n_attr:attribute num
# @return metrics of MAUC
def oahoBPClassifier(trainSet,testSet,n_class,n_attr):
    # 分解处理数据
    # data = np.loadtxt('dataset/contraceptive-5-1tra.dat', dtype=float, delimiter=', ')
    # testData = np.loadtxt('dataset/contraceptive-5-1tst.dat', dtype=float, delimiter=', ')
    tra_oaho_class, order_label = decomposeOAHO(trainSet, n_attr+1, n_class)
    x_tst, y_tst = np.split(testSet, (n_attr,), axis=1)

    x_train = []
    y_train = []

    for i in range(len(tra_oaho_class)):
        x, y = np.split(tra_oaho_class[i], (n_attr,), axis=1)
        x_train.append(x)
        y_train.append(y)
    # one-hidden layer neural networks trained with feed-forward backpropagation(BP) learning algorithm.
    # hidden nodes ranging from 3 to 30,
    clf_oaho = []
    for i in range(len(tra_oaho_class)):
        clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1)  # 15 or 20
        clf_oaho.append(clf)
        clf_oaho[i].fit(x_train[i], y_train[i])
        # clf.fit(x_train[i],y_train[i])
        # clf_oaho.append(clf)

    # for i in range(len(clf_oaho)):
    #     print clf_oaho[i].score(x_train[i], y_train[i])

    y_pred = []
    y_pred_proba = []

    for i in clf_oaho:
        y_pred.append(i.predict(x_tst))
        y_pred_proba.append(i.predict_proba(x_tst))
    # Integrated Prediction Decision(IPD)
    y_pred_final = [0] * len(y_pred[0])
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            if y_pred[i][j] != 0 and y_pred_final[j] == 0:
                y_pred_final[j] = y_pred[i][j]

    y_test = []
    for i in range(len(y_tst)):
        y_test.append(int(y_tst[i][0]))
    print y_test
    print y_pred_final
    # print statAUC(3, y_test, y_pred_final)
    mauc = statAUC(n_class, y_test, y_pred_final)
    return mauc
# data = np.loadtxt('dataset/yeast.data', dtype=str, delimiter='  ')
# id,tmp = np.split(data,(1,),axis=1)
# print tmp
# dat = tmp[0:,0:].astype(float)
# x,y = np.split(dat, (8,), axis=1)
# kf=KFold(n_splits=10,shuffle=True)    #分成几个组
# kf.get_n_splits()
# print(kf)
# total = 0
# for i_train,i_test in kf.split(x,y):
#     x_train,x_test = x[i_train],x[i_test]
#     y_train, y_test = y[i_train], y[i_test]
#     trainSet = np.append(x_train,y_train,axis=1)
#     testSet = np.append(x_test,y_test,axis=1)
#     total += oahoBPClassifier(trainSet, testSet, 10, 8)
#     print '================================================'
# print total/10