# -*- coding: UTF-8 -*-
import ensemble
from statAUC import statAUC
from sampling import *
from sklearn import tree
from decomposeOVO import decomposeOVO
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
# 分解处理数据
data = np.loadtxt('dataset/contraceptive-5-5tra.dat', dtype=float,delimiter=', ')
testData=np.loadtxt('dataset/contraceptive-5-5tst.dat',dtype=float,delimiter=', ')
# testSingleClass = decomposeOVO(testData,10,3)
tra_single_class = decomposeOVO(data,10,3)
x_tst, y_tst = np.split(testData, (9,), axis=1)

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
for i in range(len(binary_class_list)):
    x, yi = np.split(binary_class_list[i], (9,), axis=1)
    x_train.append(x)
#对每个分解的OVO构造模型
y_pred_test = []
ctree = tree.DecisionTreeClassifier()
clf = []

for i in range(len(binary_class_list)):
    temp_pool = ensemble.UnderBagging(ctree, 40)
    clf.append(temp_pool)
    clf[i].fit(x_train[i],y_train[i])
    y_pred_test.append(clf[i].predict(x_tst))
#将分类的数据结果改为多类形式并投票
y_test_temp= [([0] * len(y_pred_test)) for i in range(len(y_pred_test[0]))]
for i in range(len(y_pred_test)):
    for j in range(len(y_pred_test[i])):
        if(i==0):
            if(y_pred_test[i][j]==0):
                y_pred_test[i][j] = 1
            else:
                y_pred_test[i][j] = 2
        if(i == 1):
            if (y_pred_test[i][j] == 0):
                y_pred_test[i][j] = 1
            else:
                y_pred_test[i][j] = 3
        if(i == 2):
            if (y_pred_test[i][j] == 0):
                y_pred_test[i][j] = 2
            else:
                y_pred_test[i][j] = 3
        y_test_temp[j][i] = y_pred_test[i][j]

y_pred_final=[]
for i in y_test_temp:
    count = np.bincount(i)
    y_pred_final.append(count.argmax())
# print y_pred_final

# print metrics.accuracy_score(y_tst,y_pred_final)
# print metrics.f1_score(y_tst, y_pred_final,average='macro')

y_test=[]
for i in range(len(y_tst)):
    y_test.append(int(y_tst[i][0]))
print statAUC(3,y_test,y_pred_final)