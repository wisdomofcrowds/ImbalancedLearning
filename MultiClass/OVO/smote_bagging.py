# -*- coding: UTF-8 -*-
from brew.generation import SmoteBagging
from decomposeOVO import decomposeOVO
import numpy as np
from sklearn import metrics
from sklearn import tree
from statAUC import statAUC
from sklearn.model_selection import KFold
from decomposeOVO import changeClassLabel

# @artical:Combining One-vs-One Decomposition and Ensemble Learning for Multi-class Imbalanced Data
# @param trainSet:
# @param testSet:
# @param n_class: class num
# @param n_attr:attribute num
# @return metrics of MAUC
def smoteBaggingClassifier(trainSet,testSet,n_class,n_attr):
    # data = np.loadtxt('dataset/contraceptive-5-5tra.dat', dtype=float, delimiter=', ')
    # testData = np.loadtxt('dataset/contraceptive-5-5tst.dat', dtype=float, delimiter=', ')
    # testSingleClass = decomposeOVO(testData,10,3)
    tra_single_class = decomposeOVO(trainSet, n_attr+1, n_class)
    x_tst, y_tst = np.split(testSet, (n_attr,), axis=1)

    # connect each single class by two as binary
    binary_class_list = []
    y_train = []
    for i in range(len(tra_single_class)):
        for j in range(len(tra_single_class)):
            if (j > i):
                temp = np.empty(shape=[0, n_attr])
                temp = np.append(tra_single_class[i], tra_single_class[j], axis=0)
                binary_class_list.append(temp)
                tempy1 = 0 * np.ones((len(tra_single_class[i]),), dtype='int64')
                tempy2 = 1 * np.ones((len(tra_single_class[j]),), dtype='int64')
                y_train.append(np.concatenate((tempy1, tempy2)))
    # testBinClass = []
    # for i in range(len(testSingleClass)):
    #     for j in range(len(testSingleClass)):
    #         if(j>i):
    #             temp = np.empty(shape=[0, 9])
    #             temp = np.append(testSingleClass[i], testSingleClass[j], axis=0)
    #             testBinClass.append(temp)
    x_train = []
    # x_test=[]
    # y_test= []
    for i in range(len(binary_class_list)):
        x, yi = np.split(binary_class_list[i], (n_attr,), axis=1)
        x_train.append(x)
        # x_tst, y_tst = np.split(testBinClass[i], (9,), axis=1)
        # x_test.append(x_tst)
        # y_test.append(y_tst)
    ctree = tree.DecisionTreeClassifier()
    pool = []
    y_pred_test = []

    for i in range(len(binary_class_list)):
        temp_pool = SmoteBagging(base_classifier=ctree, n_classifiers=40, k=5)
        pool.append(temp_pool)
        # print x_train[i]
        # print y_train[i]
        pool[i].fit(x_train[i], y_train[i])
        y_pred_test.append(pool[i].predict(x_tst))
    y_test_temp = changeClassLabel(y_pred_test)
    y_pred_final = []
    for i in y_test_temp:
        count = np.bincount(i)
        y_pred_final.append(count.argmax())
    # print y_pred_final
    # print metrics.accuracy_score(y_tst, y_pred_final)
    # print metrics.roc_auc_score(y_tst, y_pred_final)
    # print metrics.f1_score(y_tst, y_pred_final, average='macro')
    y_test = []
    for i in range(len(y_tst)):
        y_test.append(int(y_tst[i][0]))
    # print statAUC(3, y_test, y_pred_final)
    print y_pred_final
    print y_test
    mauc = statAUC(3, y_test, y_pred_final)
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
#     total += smoteBaggingClassifier(trainSet, testSet, 10, 8)
#     print '================================================'
# print total/10