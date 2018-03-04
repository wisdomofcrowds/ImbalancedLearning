# -*- coding: UTF-8 -*-
from decomposeOVO import *
from decomposeOVA import decomposeOVA
import numpy as np
from sklearn.naive_bayes import GaussianNB
import heapq
from sklearn import metrics
from statAUC import statAUC
from sklearn.model_selection import KFold

# @artical:Multi-Class Pattern Classification in Imbalanced Data
# @param trainSet:
# @param testSet:
# @param n_class: class num
# @param n_attr:attribute num
# @return metrics of MAUC
def prmBayesClassifier(trainSet,testSet,n_class,n_attr):
    # 分解处理数据
    # data = np.loadtxt('dataset/contraceptive-5-1tra.dat', dtype=float, delimiter=', ')
    # testData = np.loadtxt('dataset/contraceptive-5-1tst.dat', dtype=float, delimiter=', ')
    # testSingleClass = decomposeOVO(testData,10,3)
    tra_ovo_class = decomposeOVO(trainSet, n_attr+1, n_class)
    tra_ova_class = decomposeOVA(trainSet, n_attr+1, n_class)
    x_tst, y_tst = np.split(testSet, (n_attr,), axis=1)

    # build training set for OVA
    OVA_class_list = []
    y_train_ova = []
    x_train_ova = []
    for i in range(len(tra_ova_class)):
        x, y = np.split(tra_ova_class[i], (n_attr,), axis=1)
        x_train_ova.append(x)
        y_train_ova.append(y)

    # #对每个分解的OVA构造模型
    clf_ova = []
    for i in range(len(tra_ova_class)):
        gnb = GaussianNB()
        clf_ova.append(gnb)
        clf_ova[i].fit(x_train_ova[i], y_train_ova[i])

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
                # tempy1 = 0 * np.ones((len(tra_ovo_class[i]),), dtype='int64')
                # tempy2 = 1 * np.ones((len(tra_ovo_class[j]),), dtype='int64')
                # y_train.append(np.concatenate((tempy1, tempy2)))

    for i in range(len(binary_class_list)):
        x, y = np.split(binary_class_list[i], (n_attr,), axis=1)
        x_train_ovo.append(x)
        y_train_ovo.append(y)

    # #对每个分解的OVO构造模型
    clf_ovo = []
    for i in range(len(binary_class_list)):
        gnb = GaussianNB()
        clf_ovo.append(gnb)
        clf_ovo[i].fit(x_train_ovo[i], y_train_ovo[i])
        # y_pred_tra = clf_ovo[i].predict(x_train_ovo[i])
        # y_pred_test.append(clf_ovo[i].predict(x_tst))

    # 对于testSet 利用OVA找出second best class，再用OVO确定最后的class
    y_pred_ova = []
    y_pred_proba = []
    for i in clf_ova:
        y_pred_ova.append(i.predict(x_tst))
        y_pred_proba.append(i.predict_proba(x_tst))
    y_pred_temp_list = [([0] * len(y_pred_ova)) for i in range(len(y_pred_ova[0]))]
    y_pred_proba_list = [([0] * len(y_pred_ova)) for i in range(len(y_pred_ova[0]))]
    for i in range(len(y_pred_ova)):
        for j in range(len(y_pred_ova[i])):
            y_pred_temp_list[j][i] = y_pred_ova[i][j]
            y_pred_proba_list[j][i] = y_pred_proba[i][j]
    y_pred_final = [0] * len(y_pred_temp_list)
    for i in range(len(y_pred_temp_list)):
        count = np.bincount(y_pred_temp_list[i])
        if len(count) == 1:
            y_pred_ovo = []
            for k in range(len(clf_ovo)):
                pred = clf_ovo[k].predict([x_tst[i]])
                y_pred_ovo.append(int(pred[0]))
            majorvote = np.bincount(y_pred_ovo)
            # print y_pred_ovo
            y_pred_final[i] = majorvote.argmax()
            # print majorvote.argmax()#利用ovo求最后结果
        else:
            if count[1] == 1:
                for j in range(len(y_pred_temp_list[i])):
                    if (y_pred_temp_list[i][j] == 1.0):
                        y_pred_final[i] = j + 1
                        # print j + 1
            elif count[1] == 2:
                c = []
                for j in range(len(y_pred_temp_list[i])):
                    if (y_pred_temp_list[i][j] == 1.0):
                        c.append(j + 1)
                clf = classiferForTwo(c[0], c[1], clf_ovo)
                pred = clf.predict([x_tst[i]])
                y_pred_final[i] = int(pred[0])
            else:
                c = []
                c_probe = [0] * len(y_pred_temp_list[i])
                for j in range(len(y_pred_temp_list[i])):
                    if (y_pred_temp_list[i][j] == 1.0):
                        c_probe[j] = y_pred_proba_list[i][j][1]
                tempData = heapq.nlargest(2, enumerate(c_probe), key=lambda x: x[1])
                index, vals = zip(*tempData)
                # print index
                # print vals
                if index[0] > index[1]:
                    clf = classiferForTwo(index[1] + 1, index[0] + 1, clf_ovo)
                else:
                    clf = classiferForTwo(index[0] + 1, index[1] + 1, clf_ovo)
                pred = clf.predict([x_tst[i]])
                y_pred_final[i] = int(pred[0])
                # print pred

        # print y_pred_temp_list[i]
        # print y_pred_proba_list[i]
        # print count

    # print metrics.accuracy_score(y_tst,y_pred_final)
    # print metrics.f1_score(y_tst, y_pred_final,average='macro')

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
#     total += prmBayesClassifier(trainSet, testSet, 10, 8)
#     print '================================================'
# print total/10




