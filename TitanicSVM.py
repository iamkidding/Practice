import numpy as np

def loadData():
    trainSet = open("D:/DataSet/Kaggle/Titanic/train.csv")
    trainMat = []
    trainLabel = []
    for line in trainSet:
        curLine = line.strip().split('\n')
        for linekid in curLine:
            lineProcessing = linekid.split(',')
            if lineProcessing[5] == 'male':
                lineProcessing[5] = 1.0
            else:
                lineProcessing[5] = 0.0
            trainMat.append([lineProcessing[0], lineProcessing[2], lineProcessing[5],
                            lineProcessing[6], lineProcessing[7], lineProcessing[8], lineProcessing[10]])
            trainLabel.append(lineProcessing[1])
    trainMat = trainMat[1:892]
    trainLabel = trainLabel[1:892]
    ageMale = 0.0; ageFemale = 0.0
    numMale = 0; numFemale = 0
    for line in trainMat:   #计算男女乘客年龄的平均值,男性乘客平均年龄30，女性28
        if line[3] != '' and line[2] == 1.0:
            ageMale += float(line[3])
            numMale += 1
        elif line[3] != '' and line[2] == 0.0:
            ageFemale += float(line[3])
            numFemale += 1
    ageMale /= numMale
    ageFemale /= numFemale
    for line in trainMat:
        if line[3] == '' and line[2] == 1.0:
            line[3] = ageMale
        elif line[3] == '' and line[2] == 0.0:
            line[3] = ageFemale
    for i in range(len(trainMat)):
        trainLabel[i] = float(trainLabel[i])
        for j in range(len(trainMat[0])):
            trainMat[i][j] = float(trainMat[i][j])
    print(trainMat)

loadData()