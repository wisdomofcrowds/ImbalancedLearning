# -*- coding: UTF-8 -*-
from decomposeOVO import *
from decomposeOVA import decomposeOVA
import numpy as np
from sklearn.naive_bayes import GaussianNB
import heapq
from sklearn import metrics
from statAUC import statAUC
# 分解处理数据
data = np.loadtxt('dataset/contraceptive-5-1tra.dat', dtype=float,delimiter=', ')
testData=np.loadtxt('dataset/contraceptive-5-1tst.dat',dtype=float,delimiter=', ')
# testSingleClass = decomposeOVO(testData,10,3)
tra_ovo_class = decomposeOVO(data,10,3)
tra_ova_class = decomposeOVA(data,10,3)
x_tst, y_tst = np.split(testData, (9,), axis=1)

# build training set for OVA
OVA_class_list = []
y_train_ova = []
x_train_ova = []
for i in range(len(tra_ova_class)):
    x,y = np.split(tra_ova_class[i], (9,), axis=1)
    x_train_ova.append(x)
    y_train_ova.append(y)

# #对每个分解的OVA构造模型
clf_ova = []
for i in range(len(tra_ova_class)):
    gnb = GaussianNB()
    clf_ova.append(gnb)
    clf_ova[i].fit(x_train_ova[i],y_train_ova[i])


#connect each single class by two as binary
binary_class_list=[]
x_train_ovo=[]
y_train_ovo=[]
for i in range(len(tra_ovo_class)):
    for j in range(len(tra_ovo_class)):
        if(j>i):
            temp = np.empty(shape=[0, 9])
            temp = np.append(tra_ovo_class[i], tra_ovo_class[j], axis=0)
            binary_class_list.append(temp)
            # tempy1 = 0 * np.ones((len(tra_ovo_class[i]),), dtype='int64')
            # tempy2 = 1 * np.ones((len(tra_ovo_class[j]),), dtype='int64')
            # y_train.append(np.concatenate((tempy1, tempy2)))

for i in range(len(binary_class_list)):
    x, y = np.split(binary_class_list[i], (9,), axis=1)
    x_train_ovo.append(x)
    y_train_ovo.append(y)

# #对每个分解的OVO构造模型
clf_ovo = []
for i in range(len(binary_class_list)):
    gnb = GaussianNB()
    clf_ovo.append(gnb)
    clf_ovo[i].fit(x_train_ovo[i],y_train_ovo[i])
    # y_pred_tra = clf_ovo[i].predict(x_train_ovo[i])
    # y_pred_test.append(clf_ovo[i].predict(x_tst))

#对于testSet 利用OVA找出second best class，再用OVO确定最后的class
y_pred_ova=[]
y_pred_proba=[]
for i in clf_ova:
    y_pred_ova.append(i.predict(x_tst))
    y_pred_proba.append(i.predict_proba(x_tst))
y_pred_temp_list = [([0] * len(y_pred_ova)) for i in range(len(y_pred_ova[0]))]
y_pred_proba_list=[([0] * len(y_pred_ova)) for i in range(len(y_pred_ova[0]))]
for i in range(len(y_pred_ova)):
    for j in range(len(y_pred_ova[i])):
        y_pred_temp_list[j][i]=y_pred_ova[i][j]
        y_pred_proba_list[j][i]=y_pred_proba[i][j]
y_pred_final = [0]*len(y_pred_temp_list)
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
        if count[1]==1:
            for j in range(len(y_pred_temp_list[i])):
                if(y_pred_temp_list[i][j]==1.0):
                    y_pred_final[i]=j+1
                    # print j + 1
        elif count[1]==2:
            c = []
            for j in range(len(y_pred_temp_list[i])):
                if(y_pred_temp_list[i][j]==1.0):
                    c.append(j+1)
            clf = classiferForTwo(c[0],c[1],clf_ovo)
            pred=clf.predict([x_tst[i]])
            y_pred_final[i]=int(pred[0])
        else:
            c = []
            c_probe = [0]*len(y_pred_temp_list[i])
            for j in range(len(y_pred_temp_list[i])):
                if (y_pred_temp_list[i][j] == 1.0):
                    c_probe[j] = y_pred_proba_list[i][j][1]
            tempData = heapq.nlargest(2,enumerate(c_probe),key=lambda x:x[1])
            index,vals=zip(*tempData)
            # print index
            # print vals
            if index[0]>index[1]:
                clf= classiferForTwo(index[1]+1,index[0]+1,clf_ovo)
            else:
                clf = classiferForTwo(index[0] + 1, index[1] + 1, clf_ovo)
            pred = clf.predict([x_tst[i]])
            y_pred_final[i]=int(pred[0])
            # print pred

    # print y_pred_temp_list[i]
    # print y_pred_proba_list[i]
    # print count



# print metrics.accuracy_score(y_tst,y_pred_final)
# print metrics.f1_score(y_tst, y_pred_final,average='macro')

y_test=[]
for i in range(len(y_tst)):
    y_test.append(int(y_tst[i][0]))
print statAUC(3,y_test,y_pred_final)



