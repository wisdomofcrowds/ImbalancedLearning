# -*- coding: UTF-8 -*-
from statAUC import statAUC
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from decomposeOVO import decomposeOVO
import numpy as np
from sklearn.model_selection import KFold

# @artical:Dynamic classifier selection for One-vs-One strategy: Avoiding non-competent classifiers
# @param trainSet:
# @param testSet:
# @param n_class: class num
# @param n_attr:attribute num
# @return metrics of MAUC
def dynamicOVOClassifier(trainSet,testSet,n_class,n_attr):
    # data = np.loadtxt('dataset/contraceptive-5-5tra.dat', dtype=float, delimiter=', ')
    # testData = np.loadtxt('dataset/contraceptive-5-5tst.dat', dtype=float, delimiter=', ')
    X_train, Y_train = np.split(trainSet, (n_attr,), axis=1)
    tra_ovo_class = decomposeOVO(trainSet, n_attr+1, n_class)
    for i in tra_ovo_class:
        print i
    x_tst, y_tst = np.split(testSet, (n_attr,), axis=1)

    # connect each single class by two as binary
    binary_class_list = []
    x_train_ovo = []
    y_train_ovo = []
    for i in range(len(tra_ovo_class)):
        for j in range(len(tra_ovo_class)):
            if (j > i):
                temp = np.empty(shape=[0, n_attr])
                temp = np.append(tra_ovo_class[i], tra_ovo_class[j], axis=0)
                binary_class_list.append(temp)

    for i in range(len(binary_class_list)):
        x, y = np.split(binary_class_list[i], (n_attr,), axis=1)
        x_train_ovo.append(x)
        y_train_ovo.append(y)

    # 构造 KNN 模型
    K_neighbors = 3*n_class
    neigh = NearestNeighbors(n_neighbors=K_neighbors)
    neigh.fit(X_train)

    # #对每个分解的OVO构造模型
    clf_ovo = []
    for i in range(len(binary_class_list)):
        clf = GaussianNB()
        # clf = tree.DecisionTreeClassifier()
        # clf = svm.SVC(C=1.0, kernel='rbf', gamma=20, decision_function_shape='ovo',probability=True)
        clf.fit(x_train_ovo[i], y_train_ovo[i])
        clf_ovo.append(clf)

    R_dyn_list = []
    # for i in range(len(x_tst)):
    #     for j in range(len(clf_ovo)):
    #         print 'x_tst ', i, 'clf_ovo ', j, ' ', clf_ovo[j].predict_proba([x_tst[i]]), ' ', clf_ovo[j].predict([x_tst[i]])
    multiClass = range(1,n_class+1)
    # 构造动态OVO矩阵
    for i in range(len(x_tst)):
        # find neighbors
        nbrs = neigh.kneighbors([x_tst[i]])
        nbrs_index = nbrs[1][0]
        neigh_Y = []
        for index in nbrs_index:
            neigh_Y.append(Y_train[index][0])
        count = {k: neigh_Y.count(k) for k in set(neigh_Y)}
        if len(count) == 1:
            nbrs = neigh.kneighbors([x_tst[i]], n_neighbors=2 * K_neighbors)
            nbrs_index = nbrs[1][0]
            neigh_Y = []
            for index in nbrs_index:
                neigh_Y.append(Y_train[index][0])
            count = {k: neigh_Y.count(k) for k in set(neigh_Y)}

        non_comptent_class = list(set(multiClass).difference(set(count)))
        print count.keys()

        R_dyn = [([0] * n_class) for r in range(n_class)]
        print R_dyn
        # for j in range(len(clf_ovo)):
        # pred_proba = clf_ovo[j].predict_proba([x_tst[i]])
        # print 'x_tst ', i, 'clf_ovo ', j, ' ', clf_ovo[j].predict_proba([x_tst[i]]), ' ', clf_ovo[j].predict([x_tst[i]])
        j = 0
        for m in range(0, n_class):
            for n in range(0, n_class):
                if n > m:
                    pred_proba = clf_ovo[j].predict_proba([x_tst[i]])
                    # print 'x_tst ', i, 'clf_ovo ', j, ' ', clf_ovo[j].predict_proba([x_tst[i]]), ' ', clf_ovo[
                    #     j].predict(
                    #     [x_tst[i]])
                    j += 1
                    if len(non_comptent_class) == 0:
                        R_dyn[m][n] = pred_proba[0][0]
                        R_dyn[n][m] = pred_proba[0][1]
                    else:
                        for a in non_comptent_class:
                            # print a, m
                            # print a, n
                            if n != a - 1 and m != a - 1:
                                R_dyn[m][n] = pred_proba[0][0]
                                R_dyn[n][m] = pred_proba[0][1]
        print R_dyn
        R_sum = []
        for row in R_dyn:
            R_sum.append(sum(row))
        R_dyn_list.append(R_sum)
        print
    y_pred_final = []
    for i in range(len(R_dyn_list)):
        print R_dyn_list[i]
        temp = R_dyn_list[i].index(max(R_dyn_list[i])) + 1
        y_pred_final.append(temp)

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
# print dat
# #data = np.loadtxt('dataset/glass.dat', dtype=float, delimiter=', ')
# x,y = np.split(dat, (8,), axis=1)
# kf=KFold(n_splits=10,shuffle=True)    #分成几个组
# kf.get_n_splits()
# print(kf)
# total = 0
# for i_train,i_test in kf.split(x,y):
#     x_train,x_test = x[i_train],x[i_test]
#     y_train, y_test = y[i_train], y[i_test]
#     for i in y_test:
#         print i
#     trainSet = np.append(x_train,y_train,axis=1)
#     testSet = np.append(x_test,y_test,axis=1)
#     total += dynamicOVOClassifier(trainSet, testSet, 10, 8)
#     print '================================================'
# print total/10
