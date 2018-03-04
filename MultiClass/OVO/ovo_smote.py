# -*- coding: UTF-8 -*-
from statAUC import statAUC
from sampling import *
from sklearn import tree
from decomposeOVO import decomposeOVO
import numpy as np
from smote import SMOTE
from sklearn.model_selection import KFold

# @artical:Multi-class Imbalanced Data-Sets with Linguistic Fuzzy Rule Based Classification Systems Based on Pairwise Learning
# @param trainSet:
# @param testSet:
# @param n_class: class num
# @param n_attr:attribute num
# @return metrics of MAUC
def ovoSmoteClassifier(trainSet,testSet,n_class,n_attr):
    # data = np.loadtxt('dataset/contraceptive-5-5tra.dat', dtype=float, delimiter=', ')
    # testData = np.loadtxt('dataset/contraceptive-5-5tst.dat', dtype=float, delimiter=', ')
    tra_ovo_class = decomposeOVO(trainSet, n_attr+1, n_class)

    # for i in tra_ovo_class:
    #     print len(i)
    x_tst, y_tst = np.split(testSet, (n_attr,), axis=1)

    # connect each single class by two as binary
    binary_class_list = []
    # binary_class_IR=[]
    x_train_ovo = []
    y_train_ovo = []
    for i in range(len(tra_ovo_class)):
        for j in range(len(tra_ovo_class)):
            k_neigh = 5
            if (j > i):
                ciSize = float(len(tra_ovo_class[i]))
                cjSize = float(len(tra_ovo_class[j]))
                if ciSize < k_neigh:
                    k_neigh = int(ciSize)
                if cjSize < k_neigh:
                    k_neigh = int(cjSize)
                syntheticSamples = []
                print ciSize, ' ', cjSize
                binary_class_IR = 0
                if ciSize > cjSize:
                    binary_class_IR = ciSize / cjSize
                    if binary_class_IR > 1.5:
                        print int((binary_class_IR - 1)) * 100
                        syntheticSamples = SMOTE(tra_ovo_class[j], int((binary_class_IR - 1)) * 100, k_neigh)
                else:
                    binary_class_IR = cjSize / ciSize
                    if binary_class_IR > 1.5:
                        print int((binary_class_IR - 1)) * 100
                        syntheticSamples = SMOTE(tra_ovo_class[i], int((binary_class_IR - 1)) * 100, k_neigh)
                temp = np.empty(shape=[0, n_attr])
                temp = np.append(tra_ovo_class[i], tra_ovo_class[j], axis=0)
                if len(syntheticSamples) > 0:
                    temp = np.append(temp, syntheticSamples, axis=0)
                binary_class_list.append(temp)

    for i in range(len(binary_class_list)):
        # print len(binary_class_list[i])
        x, y = np.split(binary_class_list[i], (n_attr,), axis=1)
        x_train_ovo.append(x)
        y_train_ovo.append(y)

    clf_ovo = []
    y_pred_tst = []
    for i in range(len(binary_class_list)):
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train_ovo[i], y_train_ovo[i])
        y_pred_tst.append(clf.predict(x_tst))
        # print clf.score(x_train_ovo[i], y_train_ovo[i])
        clf_ovo.append(clf)
    y_pred_temp = [([0] * len(y_pred_tst)) for i in range(len(y_pred_tst[0]))]
    for i in range(len(y_pred_tst)):
        for j in range(len(y_pred_tst[0])):
            y_pred_temp[j][i] = y_pred_tst[i][j]

    y_pred_final = []
    for i in y_pred_temp:
        count = np.bincount(i)
        y_pred_final.append(count.argmax())

    # print np.bincount(y_pred_final)


    y_test = []
    for i in range(len(y_tst)):
        y_test.append(int(y_tst[i][0]))
    # print statAUC(3, y_test, y_pred_final)
    print y_pred_final
    print y_test
    mauc = statAUC(n_class, y_test, y_pred_final)
    return mauc
# data = np.loadtxt('dataset/yeast.data', dtype=str, delimiter='  ')
# id,tmp = np.split(data,(1,),axis=1)
# print tmp
# dat = tmp[0:,0:].astype(float)
# print dat
# #data = np.loadtxt('dataset/zz_glass.dat', dtype=float, delimiter=', ')
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
#     total += ovoSmoteClassifier(trainSet, testSet, 10, 8)
#     print '================================================'
# print total/10




