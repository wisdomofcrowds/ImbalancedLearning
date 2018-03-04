import numpy as np
import heapq
from decomposeOVO import decomposeOVO
def decomposeOAHO(data,n_col,n_class):
    ovo_class = decomposeOVO(data, n_col, n_class)
    label_index = n_col - 1
    decompose_class = []
    count = [0]* (n_class+1)
    for i in range(len(ovo_class)):
        count[i+1]=len(ovo_class[i])
    temp = heapq.nlargest(n_class, enumerate(count), key=lambda x: x[1])
    index,vals = zip(*temp)
    for i in range(0,len(index)-1):
        if(i != (len(index)-2)):
            negative = np.empty(shape=[0,n_col])
            for j in range(i+1,len(index)):
                #print index[j]-1
                negative = np.append(negative,ovo_class[index[j]-1],axis=0)

            for j in range(len(negative)):
                # print negative[j][label_index],index[i]
                if(negative[j][label_index]!= index[i]):
                    negative[j][label_index] = 0
            temp = np.append(ovo_class[index[i]-1],negative,axis=0)
            decompose_class.append(temp)

        else:
            temp = np.append(ovo_class[index[i] - 1], ovo_class[index[i+1] - 1], axis=0)
            decompose_class.append(temp)
    order_label=[]
    for i in range(len(decompose_class)):
        order_label.append(decompose_class[i][0][label_index])
    order_label.append(decompose_class[-1][-1][-1])
    return decompose_class,order_label

    # for i in decompose_class:
    #     print len(i)
    # print index
    # print vals

#
# data = np.loadtxt('dataset/contraceptive-5-1tra.dat', dtype=float,delimiter=', ')
# decomposeOAHO(data,10,7)