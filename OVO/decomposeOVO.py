import numpy as np

def decomposeOVO(data,n_col,n_class):
    decompose_class=[]
    label_index = n_col-1
    for i in range(1, n_class+1):
        temp = np.empty(shape=[0, n_col])
        for j in range(len(data)):
            if data[j][label_index] == i:
                temp = np.append(temp, np.array([data[j]]), axis=0)
        decompose_class.append(temp)

    return decompose_class
# data = np.loadtxt('dataset/contraceptive-5-1tra.dat', dtype=float,delimiter=', ')
# ovo = decomposeOVO(data,10,7)
# print ovo

def classiferForTwo(c1,c2,model):
    modelNum=len(model)
    if modelNum == 3:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        else:
            return model[2]

    # if modelNum == 4:



