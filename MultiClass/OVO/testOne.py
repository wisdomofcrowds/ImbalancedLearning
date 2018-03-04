# -*- coding: UTF-8 -*-
from smote_bagging import smoteBaggingClassifier
from under_bagging import underBaggingClassifier
from prm_bayes import prmBayesClassifier
from ovo_smote import ovoSmoteClassifier
from OAHO_BP import oahoBPClassifier
from dynamicOVO import dynamicOVOClassifier
import numpy as np
from sklearn.model_selection import KFold
def panbased_label(s):
    it = {'0': 1,'1':2,'2':3,'3':4,'4':5,'5':6,'6':7,'7':8,'8':9,'9':10}
    return it[s]

def balance_label(s):
    it = {'L': 1, 'B': 2, 'R': 3}
    return it[s]
def winered_label(s):
    it = {'3': 1,'4':2,'5':3,'6':4,'7':5,'8':6}
    return it[s]
def winewhite_label(s):
    it = {'3': 1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7}
    return it[s]
def cleveland_label(s):
    it = {'0': 1,'1':2,'2':3,'3':4,'4':5}
    return it[s]
finalResult = [0]*6
fw = open('Result/shuttleResult.txt', 'a+')
print>>fw,'dataset:shuttle'
for times in range(0,10):
    #balance class = 3 attr = 4
    # data = np.loadtxt('dataset/balance.dat', dtype=float, delimiter=', ', converters={4: balance_label})
    # x,y = np.split(data, (4,), axis=1)

    # yeast class 10 attr 8
    # data = np.loadtxt('dataset/yeast.data', dtype=str, delimiter='  ')
    # id, tmp = np.split(data, (1,), axis=1)
    # dat = tmp[0:, 0:].astype(float)
    # x, y = np.split(dat, (8,), axis=1)

    # pageblocks class 5 attr 10
    # data = np.loadtxt('dataset/page-blocks.dat', dtype=float, delimiter=', ')
    # x,y = np.split(data, (10,), axis=1)

    # cleveland class 5 attr 13
    # data = np.loadtxt('dataset/cleveland.dat', dtype=float, delimiter=',',converters={13:cleveland_label})
    # x,y = np.split(data, (13,), axis=1)

    # shuttle class 7 attr 9
    data = np.loadtxt('dataset/shuttle1.dat', dtype=float, delimiter=',')
    x,y = np.split(data, (9,), axis=1)

    kf=KFold(n_splits=10,shuffle=True)    #分成几个组
    kf.get_n_splits()
    print(kf)
    n_class = 7
    n_attr = 9
    index = 0
    result = [0]*6
    print result
    for i_train,i_test in kf.split(x,y):
        x_train,x_test = x[i_train],x[i_test]
        y_train, y_test = y[i_train], y[i_test]
        trainSet = np.append(x_train,y_train,axis=1)
        testSet = np.append(x_test,y_test,axis=1)
        result[0] += smoteBaggingClassifier(trainSet, testSet, n_class, n_attr)
        result[1] += underBaggingClassifier(trainSet, testSet, n_class, n_attr)
        result[2] += prmBayesClassifier(trainSet, testSet, n_class, n_attr)
        result[3] += oahoBPClassifier(trainSet, testSet, n_class, n_attr)
        result[4] += ovoSmoteClassifier(trainSet, testSet, n_class, n_attr)
        result[5] += dynamicOVOClassifier(trainSet, testSet, n_class, n_attr)
        print '================================================'
    print >> fw,'the ',times,'th experiment:'

    for i in range(len(result)):
        result[i]=result[i]/10
        finalResult[i]+=result[i]
        print >> fw, i ,': ',result[i]
    # finalResult.append(result)
    print 'finalResult',times,finalResult

print '------------------------------------------'
print >> fw,'--------------finalResult----------------'
for i in range(len(finalResult)):
    print finalResult[i]/10
    print >> fw,finalResult[i]/10

fw.close()