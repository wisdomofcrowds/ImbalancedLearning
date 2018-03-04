import numpy as np
from decomposeOVO import decomposeOVO
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
def led7digit_label(s):
    it = {'0': 1,'1':2,'2':3,'3':4,'4':5,'5':6,'6':7,'7':8,'8':9,'9':10}
    return it[s]

# yeast class 10 attr 8
data = np.loadtxt('dataset/yeast.data', dtype=str, delimiter='  ')
id,tmp = np.split(data,(1,),axis=1)
# print tmp
dat = tmp[0:,0:].astype(float)
# x,y = np.split(dat, (8,), axis=1)

#glass class 6 attr 9
# data = np.loadtxt('dataset/glass.data', dtype=float, delimiter=',')
# id,dat = np.split(data,(1,),axis=1)
# x,y = np.split(dat, (9,), axis=1)

#ecoli class 6 attr 7
# data = np.loadtxt('dataset/ecoli.dat', dtype=float, delimiter=',')
# x,y = np.split(data, (7,), axis=1)

#pageblocks class 5 attr 10
# data = np.loadtxt('dataset/page-blocks.dat', dtype=float, delimiter=', ')
# x,y = np.split(data, (10,), axis=1)

#new-thyroid class 3 attr 5
# data = np.loadtxt('dataset/new-thyroid.dat', dtype=float, delimiter=', ')
# x,y = np.split(data, (5,), axis=1)

#no-panbased class 10 attr 16
# data = np.loadtxt('dataset/penbased.dat', dtype=float, delimiter=', ', converters={16: panbased_label})
# x,y = np.split(data, (16,), axis=1)

#balance class 3 attr 4
# data = np.loadtxt('dataset/balance.dat', dtype=float, delimiter=', ', converters={4: balance_label})
# x,y = np.split(data, (4,), axis=1)
#contraceptive class 3 attr 9
# data = np.loadtxt('dataset/contraceptive.dat', dtype=float, delimiter=', ')
# x,y = np.split(data, (9,), axis=1)

#ZZ-wine class 3 attr 13
# data = np.loadtxt('dataset/zz_wine.dat', dtype=float, delimiter=', ')
# x,y = np.split(data, (13,), axis=1)

#shuttle class 7 attr 9
# data = np.loadtxt('dataset/shuttle1.dat', dtype=float, delimiter=',')
# x,y = np.split(data, (9,), axis=1)
# print data
# print x
# print y

#winequality-red class 6 attr 11
# data = np.loadtxt('dataset/winequality-red.csv', dtype=float, delimiter=';',converters={11: winered_label})
# x,y = np.split(data, (11,), axis=1)
# print data
# print x
# print y

#winequality-white class 7 attr 11
# data = np.loadtxt('dataset/winequality-white.csv', dtype=float, delimiter=';',converters={11:winewhite_label})
# x,y = np.split(data, (11,), axis=1)
# print data
# print x
# print y

#cleveland class 5 attr 13
# data = np.loadtxt('dataset/cleveland.dat', dtype=float, delimiter=',',converters={13:cleveland_label})
# x,y = np.split(data, (13,), axis=1)
# print data
# print x
# print y

#dermatology class 6 attr 34
# data = np.loadtxt('dataset/dermatology.dat', dtype=float, delimiter=',')
# x,y = np.split(data,(34,),axis=1)

#led7digit class 10 attr 7
data = np.loadtxt('dataset/led7digit.dat', dtype=float, delimiter=', ', converters={7: led7digit_label})
x,y = np.split(data,(7,),axis=1)
print y
#hayes-roth class 3 attr 4
# data = np.loadtxt('dataset/hayes-roth.dat', dtype=int, delimiter=', ')

#breastTissue class 6 attr 9
# data = np.loadtxt('dataset/BreastTissue.csv', dtype=float, delimiter=';')

#UserKnowledgeModelling class 4 attr 5
# data = np.loadtxt('dataset/UserKnowledgeModelling.csv', dtype=float, delimiter=';')

#vertebral_column_data class 3 attr 6
# data = np.loadtxt('dataset/column_3C.dat',dtype=float,delimiter=' ')

print len(data)
print
ovo = decomposeOVO(dat,9,10)
for i in ovo:
    print len(i)