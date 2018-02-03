from brew.generation import SmoteBagging
from decomposeOVO import decomposeOVO
import numpy as np
from sklearn import metrics
from sklearn import tree
from statAUC import statAUC
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
# testBinClass = []
# for i in range(len(testSingleClass)):
#     for j in range(len(testSingleClass)):
#         if(j>i):
#             temp = np.empty(shape=[0, 9])
#             temp = np.append(testSingleClass[i], testSingleClass[j], axis=0)
#             testBinClass.append(temp)
x_train=[]
#x_test=[]
#y_test= []
for i in range(len(binary_class_list)):
    x, yi = np.split(binary_class_list[i], (9,), axis=1)
    x_train.append(x)
    # x_tst, y_tst = np.split(testBinClass[i], (9,), axis=1)
    # x_test.append(x_tst)
    # y_test.append(y_tst)
# for i in range(0,3):
#     print x_train[i].shape
#     print y[i]
#     print x_test[i].shape
#     print y_test[i].shape
#     print


# x1,yi1 = np.split(binary_class_list[0] , (9,), axis=1 )
# x1_test,y1_test=np.split(testBinClass[0], (9,), axis=1)
#
# x2,yi2 = np.split(binary_class_list[1],(9,), axis=1)
# x2_test, y2_test = np.split(testBinClass[1], (9,), axis=1)

# print x1.shape
# print x1_test.shape

# leny1 = len(tra_single_class[0])
# leny2 = len(tra_single_class[1])
# tempy1 = 0 * np.ones((leny1,), dtype='int64')
# tempy2 = 1 * np.ones((leny2,), dtype='int64')
# y1 = np.concatenate((tempy1,tempy2))
ctree = tree.DecisionTreeClassifier()
pool = []
y_pred_test = []

# pool1 = SmoteBagging(base_classifier=ctree, n_classifiers=40, k=5)
# pool2 = SmoteBagging(base_classifier=ctree, n_classifiers=40, k=5)
# pool3 = SmoteBagging(base_classifier=ctree, n_classifiers=40, k=5)
#
# pool1.fit(x_train[0],y_train[0])
# pool2.fit(x_train[1],y_train[1])
# pool3.fit(x_train[2],y_train[2])
#
# y_pred_test.append(pool1.predict(x_tst))
# y_pred_test.append(pool2.predict(x_tst))
# y_pred_test.append(pool3.predict(x_tst))
for i in range(len(binary_class_list)):
    temp_pool = SmoteBagging(base_classifier=ctree, n_classifiers=40, k=5)
    pool.append(temp_pool)
    pool[i].fit(x_train[i],y_train[i])
    y_pred_test.append(pool[i].predict(x_tst))
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
#print y_pred_final
print metrics.accuracy_score(y_tst,y_pred_final)
#print metrics.roc_auc_score(y_tst, y_pred_final)
print metrics.f1_score(y_tst, y_pred_final, average='macro')
y_test=[]
for i in range(len(y_tst)):
    y_test.append(int(y_tst[i][0]))
print statAUC(3,y_test,y_pred_final)