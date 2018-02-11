import numpy as np
from decomposeOVO import decomposeOVO
def decomposeOVA(data,n_col,n_class):
    ovo_class = decomposeOVO(data, n_col, n_class)
    decompose_class = []
    label_index = n_col - 1
    for i in range(len(ovo_class)):
        temp = np.empty(shape=[0, n_col])
        positive = ovo_class[i]
        for j in range(len(positive)):
            positive[j][label_index] = 1
        negative = np.empty(shape=[0, n_col])
        for k in range(len(ovo_class)):
            if i != k:
                negative = np.append(negative, ovo_class[k], axis=0)
        for m in range(len(negative)):
            negative[m][label_index] = 0
        temp = np.append(positive,negative,axis=0)
        decompose_class.append(temp)
    return decompose_class
# data = np.loadtxt('dataset/contraceptive-5-5tra.dat', dtype=float,delimiter=', ')
# single_class = decomposeOVO(data,10,3)
# ova = decomposeOVA(single_class,10)
#
# print len(ova[0])
# print len(ova[1])
# print len(ova[2])


