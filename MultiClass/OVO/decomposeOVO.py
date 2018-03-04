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

def classiferForTwo(c1,c2,model):
    modelNum=len(model)
    # class = 3
    if modelNum == 3:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        else:
            return model[2]

    # class = 4
    if modelNum == 6:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        elif c1==1 and c2==4:
            return model[2]
        elif c1==2 and c2==3:
            return model[3]
        elif c1==2 and c2==4:
            return model[4]
        else:
            return model[5]

    # class = 5
    if modelNum == 10:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        elif c1==1 and c2==4:
            return model[2]
        elif c1==1 and c2==5:
            return model[3]
        elif c1==2 and c2==3:
            return model[4]
        elif c1==2 and c2==4:
            return model[5]
        elif c1==2 and c2==5:
            return model[6]
        elif c1==3 and c2==4:
            return model[7]
        elif c1==3 and c2==5:
            return model[8]
        else:
            return model[9]

    # class = 6
    if modelNum == 15:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        elif c1==1 and c2==4:
            return model[2]
        elif c1==1 and c2==5:
            return model[3]
        elif c1==1 and c2==6:
            return model[4]
        elif c1==2 and c2==3:
            return model[5]
        elif c1==2 and c2==4:
            return model[6]
        elif c1==2 and c2==5:
            return model[7]
        elif c1==2 and c2==6:
            return model[8]
        elif c1==3 and c2==4:
            return model[9]
        elif c1==3 and c2==5:
            return model[10]
        elif c1==3 and c2==6:
            return model[11]
        elif c1==4 and c2==5:
            return model[12]
        elif c1==4 and c2==6:
            return model[13]
        else:
            return model[14]

    # class = 7
    if modelNum == 21:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        elif c1==1 and c2==4:
            return model[2]
        elif c1==1 and c2==5:
            return model[3]
        elif c1==1 and c2==6:
            return model[4]
        elif c1==1 and c2==7:
            return model[5]
        elif c1==2 and c2==3:
            return model[6]
        elif c1==2 and c2==4:
            return model[7]
        elif c1==2 and c2==5:
            return model[8]
        elif c1==2 and c2==6:
            return model[9]
        elif c1==2 and c2==7:
            return model[10]
        elif c1==3 and c2==4:
            return model[11]
        elif c1==3 and c2==5:
            return model[12]
        elif c1==3 and c2==6:
            return model[13]
        elif c1==3 and c2==7:
            return model[14]
        elif c1==4 and c2==5:
            return model[15]
        elif c1==4 and c2==6:
            return model[16]
        elif c1==4 and c2==7:
            return model[17]
        elif c1==5 and c2==6:
            return model[18]
        elif c1==5 and c2==7:
            return model[19]
        else:
            return model[20]

    # class = 8
    if modelNum == 28:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        elif c1==1 and c2==4:
            return model[2]
        elif c1 ==1 and c2 ==5:
            return model[3]
        elif c1 ==1 and c2 ==6:
            return model[4]
        elif c1 ==1 and c2 ==7:
            return model[5]
        elif c1 ==1 and c2 ==8:
            return model[6]
        elif c1 ==2 and c2 ==3:
            return model[7]
        elif c1 ==2 and c2 ==4:
            return model[8]
        elif c1 ==2 and c2 ==5:
            return model[9]
        elif c1 ==2 and c2==6 :
            return model[10]
        elif c1 ==2 and c2 ==7:
            return model[11]
        elif c1 ==2 and c2==8 :
            return model[12]
        elif c1 ==3 and c2 ==4:
            return model[13]
        elif c1 ==3 and c2 ==5:
            return model[14]
        elif c1 ==3 and c2 ==6:
            return model[15]
        elif c1 ==3 and c2 ==7:
            return model[16]
        elif c1 ==3 and c2 ==8:
            return model[17]
        elif c1 ==4 and c2 ==5:
            return model[18]
        elif c1 ==4 and c2 ==6:
            return model[19]
        elif c1 ==4 and c2 ==7:
            return model[20]
        elif c1 ==4 and c2 ==8:
            return model[21]
        elif c1 ==5 and c2 ==6:
            return model[22]
        elif c1 ==5 and c2 ==7:
            return model[23]
        elif c1 ==5 and c2 ==8 :
            return model[24]
        elif c1 ==6 and c2 ==7:
            return model[25]
        elif c1 ==6 and c2 ==8:
            return model[26]
        else:
            return model[27]

    # class = 9
    if modelNum == 36:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        elif c1==1 and c2==4:
            return model[2]
        elif c1 ==1 and c2 ==5:
            return model[3]
        elif c1 ==1 and c2 ==6:
            return model[4]
        elif c1 ==1 and c2 ==7:
            return model[5]
        elif c1 ==1 and c2 ==8:
            return model[6]
        elif c1 ==1 and c2 ==9:
            return model[7]
        elif c1 ==2 and c2 ==3:
            return model[8]
        elif c1 ==2 and c2 ==4:
            return model[9]
        elif c1 ==2 and c2 ==5:
            return model[10]
        elif c1 ==2 and c2==6 :
            return model[11]
        elif c1 ==2 and c2 ==7:
            return model[12]
        elif c1 ==2 and c2==8 :
            return model[13]
        elif c1 ==2 and c2 ==9:
            return model[14]
        elif c1 ==3 and c2 ==4:
            return model[15]
        elif c1 ==3 and c2 ==5:
            return model[16]
        elif c1 ==3 and c2 ==6:
            return model[17]
        elif c1 ==3 and c2 ==7:
            return model[18]
        elif c1 ==3 and c2 ==8:
            return model[19]
        elif c1 ==3 and c2 ==9:
            return model[20]
        elif c1 ==4 and c2 ==5:
            return model[21]
        elif c1 ==4 and c2 ==6:
            return model[22]
        elif c1 ==4 and c2 ==7:
            return model[23]
        elif c1 ==4 and c2 ==8:
            return model[24]
        elif c1 ==4 and c2 ==9:
            return model[25]
        elif c1 ==5 and c2 ==6:
            return model[26]
        elif c1 ==5 and c2 ==7:
            return model[27]
        elif c1 ==5 and c2 ==8 :
            return model[28]
        elif c1 ==5 and c2 ==9:
            return model[29]
        elif c1 ==6 and c2 ==7:
            return model[30]
        elif c1 ==6 and c2 ==8:
            return model[31]
        elif c1 ==6 and c2 ==9:
            return model[32]
        elif c1 ==7 and c2 ==8:
            return model[33]
        elif c1 ==7 and c2 ==9:
            return model[34]
        else:
            return model[35]

    # class = 10
    if modelNum == 45:
        if c1==1 and c2 ==2:
            return model[0]
        elif c1==1 and c2==3:
            return model[1]
        elif c1==1 and c2==4:
            return model[2]
        elif c1 ==1 and c2 ==5:
            return model[3]
        elif c1 ==1 and c2 ==6:
            return model[4]
        elif c1 ==1 and c2 ==7:
            return model[5]
        elif c1 ==1 and c2 ==8:
            return model[6]
        elif c1 ==1 and c2 ==9:
            return model[7]
        elif c1 ==1 and c2 ==10:
            return model[8]
        elif c1 ==2 and c2 ==3:
            return model[9]
        elif c1 ==2 and c2 ==4:
            return model[10]
        elif c1 ==2 and c2 ==5:
            return model[11]
        elif c1 ==2 and c2==6 :
            return model[12]
        elif c1 ==2 and c2 ==7:
            return model[13]
        elif c1 ==2 and c2==8 :
            return model[14]
        elif c1 ==2 and c2 ==9:
            return model[15]
        elif c1 ==2 and c2 ==10:
            return model[16]
        elif c1 ==3 and c2 ==4:
            return model[17]
        elif c1 ==3 and c2 ==5:
            return model[18]
        elif c1 ==3 and c2 ==6:
            return model[19]
        elif c1 ==3 and c2 ==7:
            return model[20]
        elif c1 ==3 and c2 ==8:
            return model[21]
        elif c1 ==3 and c2 ==9:
            return model[22]
        elif c1 ==3 and c2 ==10:
            return model[23]
        elif c1 ==4 and c2 ==5:
            return model[24]
        elif c1 ==4 and c2 ==6:
            return model[25]
        elif c1 ==4 and c2 ==7:
            return model[26]
        elif c1 ==4 and c2 ==8:
            return model[27]
        elif c1 ==4 and c2 ==9:
            return model[28]
        elif c1 ==4 and c2 ==10:
            return model[29]
        elif c1 ==5 and c2 ==6:
            return model[30]
        elif c1 ==5 and c2 ==7:
            return model[31]
        elif c1 ==5 and c2 ==8 :
            return model[32]
        elif c1 ==5 and c2 ==9:
            return model[33]
        elif c1 ==5 and c2 ==10:
            return model[34]
        elif c1 ==6 and c2 ==7:
            return model[35]
        elif c1 ==6 and c2 ==8:
            return model[36]
        elif c1 ==6 and c2 ==9:
            return model[37]
        elif c1 ==6 and c2 ==10:
            return model[38]
        elif c1 ==7 and c2 ==8:
            return model[39]
        elif c1 ==7 and c2 ==9:
            return model[40]
        elif c1 ==7 and c2 ==10:
            return model[41]
        elif c1 ==8 and c2 ==9:
            return model[42]
        elif c1 ==8 and c2 ==10:
            return model[43]
        else:
            return model[44]

def changeClassLabel(y_pred_test):
    y_test_temp = [([0] * len(y_pred_test)) for i in range(len(y_pred_test[0]))]
    pairNum = len(y_pred_test)
    # class =3
    if pairNum == 3:
        for i in range(len(y_pred_test)):
            for j in range(len(y_pred_test[i])):
                if (i == 0):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 2
                if (i == 1):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 3
                if (i == 2):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 3
                y_test_temp[j][i] = y_pred_test[i][j]

    # class = 4
    if pairNum == 6:
        for i in range(len(y_pred_test)):
            for j in range(len(y_pred_test[i])):
                if (i == 0):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 2
                if (i == 1):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 3
                if (i == 2):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 4
                if (i == 3):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 3
                if (i == 4):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 4
                if (i == 5):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 4
                y_test_temp[j][i] = y_pred_test[i][j]

    # class = 5
    if pairNum == 10:
        for i in range(len(y_pred_test)):
            for j in range(len(y_pred_test[i])):
                if (i == 0):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 2
                if (i == 1):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 3
                if (i == 2):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 4
                if (i == 3):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 5
                if (i == 4):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 3
                if (i == 5):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 4
                if (i == 6):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 5
                if (i == 7):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 4
                if (i == 8):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 5
                if (i == 9):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 5
                y_test_temp[j][i] = y_pred_test[i][j]

    # class = 6
    if pairNum == 15:
        for i in range(len(y_pred_test)):
            for j in range(len(y_pred_test[i])):
                if (i == 0):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 2
                if (i == 1):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 3
                if (i == 2):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 4
                if (i == 3):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 5
                if (i == 4):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 6
                if (i == 5):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 3
                if (i == 6):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 4
                if (i == 7):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 5
                if (i == 8):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 6
                if (i == 9):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 4
                if (i == 10):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 5
                if (i == 11):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 6
                if (i == 12):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 5
                if (i == 13):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 6
                if (i == 14):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 6
                y_test_temp[j][i] = y_pred_test[i][j]

    # class = 7
    if pairNum == 21:
        for i in range(len(y_pred_test)):
            for j in range(len(y_pred_test[i])):
                if (i == 0):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 2
                if (i == 1):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 3
                if (i == 2):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 4
                if (i == 3):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 5
                if (i == 4):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 6
                if (i == 5):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 7
                if (i == 6):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 3
                if (i == 7):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 4
                if (i == 8):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 5
                if (i == 9):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 6
                if (i == 10):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 7
                if (i == 11):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 4
                if (i == 12):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 5
                if (i == 13):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 6
                if (i == 14):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 7
                if (i == 15):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 5
                if (i == 16):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 6
                if (i == 17):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 7
                if (i == 18):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 6
                if (i == 19):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 7
                if (i == 20):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 7
                y_test_temp[j][i] = y_pred_test[i][j]

    # class = 8
    if pairNum == 28:
        for i in range(len(y_pred_test)):
            for j in range(len(y_pred_test[i])):
                if (i == 0):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 2
                if (i == 1):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 3
                if (i == 2):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 4
                if (i == 3):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 5
                if (i == 4):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 6
                if (i == 5):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 7
                if (i == 6):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 8
                if (i == 7):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 3
                if (i == 8):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 4
                if (i == 9):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 5
                if (i == 10):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 6
                if (i == 11):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 7
                if (i == 12):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 8
                if (i == 13):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 4
                if (i == 14):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 5
                if (i == 15):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 6
                if (i == 16):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 7
                if (i == 17):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 8
                if (i == 18):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 5
                if (i == 19):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 6
                if (i == 20):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 7
                if (i == 21):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 8
                if (i == 22):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 6
                if (i == 23):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 7
                if (i == 24):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 8
                if (i == 25):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 7
                if (i == 26):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 8
                if (i == 27):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 7
                    else:
                        y_pred_test[i][j] = 8
                y_test_temp[j][i] = y_pred_test[i][j]

    # class = 9
    if pairNum == 36:
        for i in range(len(y_pred_test)):
            for j in range(len(y_pred_test[i])):
                if (i == 0):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 2
                if (i == 1):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 3
                if (i == 2):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 4
                if (i == 3):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 5
                if (i == 4):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 6
                if (i == 5):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 7
                if (i == 6):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 8
                if (i == 7):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 9
                if (i == 8):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 3
                if (i == 9):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 4
                if (i == 10):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 5
                if (i == 11):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 6
                if (i == 12):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 7
                if (i == 13):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 8
                if (i == 14):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 9
                if (i == 15):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 4
                if (i == 16):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 5
                if (i == 17):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 6
                if (i == 18):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 7
                if (i == 19):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 8
                if (i == 20):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 9
                if (i == 21):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 5
                if (i == 22):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 6
                if (i == 23):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 7
                if (i == 24):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 8
                if (i == 25):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 9
                if (i == 26):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 6
                if (i == 27):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 7
                if (i == 28):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 8
                if (i == 29):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 9
                if (i == 30):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 7
                if (i == 31):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 8
                if (i == 32):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 9
                if (i == 33):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 7
                    else:
                        y_pred_test[i][j] = 8
                if (i == 34):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 7
                    else:
                        y_pred_test[i][j] = 9
                if (i == 35):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 8
                    else:
                        y_pred_test[i][j] = 9
                y_test_temp[j][i] = y_pred_test[i][j]

    # class = 10
    if pairNum == 45:
        for i in range(len(y_pred_test)):
            for j in range(len(y_pred_test[i])):
                if (i == 0):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 2
                if (i == 1):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 3
                if (i == 2):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 4
                if (i == 3):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 5
                if (i == 4):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 6
                if (i == 5):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 7
                if (i == 6):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 8
                if (i == 7):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 9
                if (i == 8):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 1
                    else:
                        y_pred_test[i][j] = 10
                if (i == 9):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 3
                if (i == 10):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 4
                if (i == 11):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 5
                if (i == 12):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 6
                if (i == 13):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 7
                if (i == 14):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 8
                if (i == 15):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 9
                if (i == 16):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 2
                    else:
                        y_pred_test[i][j] = 10
                if (i == 17):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 4
                if (i == 18):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 5
                if (i == 19):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 6
                if (i == 20):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 7
                if (i == 21):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 8
                if (i == 22):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 9
                if (i == 23):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 3
                    else:
                        y_pred_test[i][j] = 10
                if (i == 24):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 5
                if (i == 25):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 6
                if (i == 26):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 7
                if (i == 27):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 8
                if (i == 28):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 9
                if (i == 29):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 4
                    else:
                        y_pred_test[i][j] = 10
                if (i == 30):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 6
                if (i == 31):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 7
                if (i == 32):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 8
                if (i == 33):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 9
                if (i == 34):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 5
                    else:
                        y_pred_test[i][j] = 10
                if (i == 35):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 7
                if (i == 36):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 8
                if (i == 37):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 9
                if (i == 38):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 6
                    else:
                        y_pred_test[i][j] = 10
                if (i == 39):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 7
                    else:
                        y_pred_test[i][j] = 8
                if (i == 40):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 7
                    else:
                        y_pred_test[i][j] = 9
                if (i == 41):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 7
                    else:
                        y_pred_test[i][j] = 10
                if (i == 42):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 8
                    else:
                        y_pred_test[i][j] = 9
                if (i == 43):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 8
                    else:
                        y_pred_test[i][j] = 10
                if (i == 44):
                    if (y_pred_test[i][j] == 0):
                        y_pred_test[i][j] = 9
                    else:
                        y_pred_test[i][j] = 10

                y_test_temp[j][i] = y_pred_test[i][j]
    return y_test_temp













