import numpy as np
from sklearn import metrics
def computeAUC(c1,c2,y_pred_test,y_test):
    reals = []
    preds = []
    for i in range(len(y_test)):
        if(((y_test[i] == c1) or (y_test[i]==c2)) and ((y_pred_test[i]==c1) or (y_pred_test[i]==c2))):
            reals.append(y_test[i])
            preds.append(y_pred_test[i])
    if(len(reals)!=len(preds)):
        print "error"
        return

    if len(reals) == 0:
        return 0
    realsB = [0]*len(reals)
    predsB = [0.]*len(preds)
    predictions = []
    dist = [0]*2
    for i in range(len(realsB)):
        if(reals[i] == c2):
            realsB[i]=1
        if(preds[i] == c2):
            predsB[i] = 1.0
            dist[0] = 0.0
            dist[1] = 1.0
        else:
            predsB[i] = 0.0
            dist[0] = 1.0
            dist[1] = 0.0
    aucArea = metrics.roc_auc_score(realsB, predsB)
    return aucArea


def statAUC(n_class,y_test,y_pred_test):

    auc = 0

    for i in range(1,n_class):
        for j in range(i+1,n_class+1):
            auc += computeAUC(i,j,y_pred_test,y_test)
    auc = (2 * auc)/float(n_class*(n_class-1))
    return auc






