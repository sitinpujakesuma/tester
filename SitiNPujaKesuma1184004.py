import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

def preparation():
    ord = OrdinalEncoder()
    info = pd.read_csv('Chapter01/dataset/CurahHujan.txt', sep=',', usecols=[0,1,2,3,4,5,6,7,8,9,10], header=None, names=['Location','MinTemp','MaxTemp','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Temp9am','Temp3pm','RainTomorrow'])
    info = info.sample(frac=1)
    info = [info.iloc[:,:10], info.iloc[:, 10:]]


    dataAttr = info.pop(0)
    dataVar = info.pop(0)
    

    length = int(len(dataVar)*0.75)

    trainVar = dataVar[:length]
    trainAttr = dataAttr[:length]

    testVar = dataVar[length:]
    testAttr = dataAttr[length:]

    return [[trainAttr, trainVar], [testAttr, testVar]]

def training(trainAttr, trainVar):
    t = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)