# -*- coding: UTF-8 -*-
import numpy as np
from brew.generation import SmoteBagging
from sklearn import svm
from decomposeOVO import decomposeOVO
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

data = np.loadtxt('dataset/contraceptive-5-1tra.dat', dtype=float,delimiter=', ')
testData=np.loadtxt('dataset/contraceptive-5-1tst.dat',dtype=float,delimiter=', ')
#x_train,y_train = np.split(data , (9,), axis=1 )
x_tst, y_tst = np.split(testData, (9,), axis=1)
tra_single_class = decomposeOVO(data,10,3)
testSingleClass = decomposeOVO(testData,10,3)
# connect each single class by two as binary
binary_class_list=[]
y_train=[]
for i in range(len(tra_single_class)):
    for j in range(len(tra_single_class)):
        if(j>i):
            temp = np.empty(shape=[0, 9])
            temp = np.append(tra_single_class[i], tra_single_class[j], axis=0)
            binary_class_list.append(temp)
            tempy1 = 0 * np.ones((len(tra_single_class[i]),), dtype='int64')
            tempy2 = 1 * np.ones((len(tra_single_class[j]),), dtype='int64')
            y_train.append(np.concatenate((tempy1, tempy2)))
x_train=[]
#x_test=[]
#y_test= []
for i in range(len(binary_class_list)):
    x, yi = np.split(binary_class_list[i], (9,), axis=1)
    x_train.append(x)
testBinClass = []
for i in range(len(testSingleClass)):
    for j in range(len(testSingleClass)):
        if(j>i):
            temp = np.empty(shape=[0, 9])
            temp = np.append(testSingleClass[i], testSingleClass[j], axis=0)
            testBinClass.append(temp)
x_test, y_test = np.split(testBinClass[2], (9,), axis=1)
ctree = tree.DecisionTreeClassifier()
pool = SmoteBagging(base_classifier=ctree, n_classifiers=40, k=5)
pool.fit(x_train[2],y_train[2])
y_pred_test=pool.predict(x_test)
print y_pred_test
for i in range(len(y_pred_test)):
    if(y_pred_test[i]==0):
        y_pred_test[i]=2
    else:
        y_pred_test[i]=3
print y_pred_test
print metrics.f1_score(y_test, y_pred_test,average='weighted')
# clf=svm.SVC(C=1.0, kernel='linear',decision_function_shape='ovo')
# clf.fit(x_train,y_train.ravel())
# y_pred_test = clf.predict(x_tst)
# print 'SVM_f1 ',
# print metrics.f1_score(y_tst, y_pred_test,average='weighted')
#
# clf1 = tree.DecisionTreeClassifier()
# clf1.fit(x_train,y_train)
# y_pred_test = clf1.predict(x_tst)
# print 'DecisionTree_f1 ',
# print metrics.f1_score(y_tst, y_pred_test,average='weighted')

