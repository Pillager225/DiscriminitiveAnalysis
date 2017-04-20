#!/usr/bin/env python3
import numpy as np

# returns a dictionary whos keys are the categories and the values are a list of the data instances in that category
def splitByCategory(data, categoryColumn):
    dataDict = {}
    cats = data[:, categoryColumn]
    datas = np.delete(data, categoryColumn, 1)
    for i in range(len(data)):
        if cats[i] not in dataDict:
            dataDict[cats[i]] = [datas[i]]
        else:
            dataDict[cats[i]].append(datas[i])
    return dataDict

# expects a list of floats
# returns the tuple (trainingData, testingData)
def getTrainTestData(data, testPercent):
    numOfTest = int(len(data)*testPercent/100.0)
    testData = []
    for i in range(numOfTest):
        testData.append(data.pop())
    return (data, testData)

# splits the data for training and testing depending on global testPercentage
# makes sure that the data is split proprotionally for each category
# returns the tuple (trainingData, testingData)
def splitData(data, categoryColumn, testPercent):
    train = {}
    test = {}
    datas = splitByCategory(data, categoryColumn)
    for cat in datas.keys():
        train[cat], test[cat] = getTrainTestData(datas[cat], testPercent)
    return train, test
