#!/usr/bin/env python3

import pandas as pd # used to download the data and parse it
import numpy as np
import math
import optparse
from Preprocessing import *

def getData(url, columnNames, shuffled = False):
    data = None
    if columnNames is None:
        data = pd.read_csv(url, header=-1)
    else:
        data = pd.read_csv(url, header=-1, names=columnNames)
    retval = np.array(data)
    if shuffled:
        np.random.shuffle(retval)
    return retval

# returns an array representing the average values for each feature of the data
# data should be an array of floats
def getAverageVector(data):
    ave = np.zeros(len(data[0]), dtype=np.float32)
    for i in range(len(data)):
        for j in range(len(data[i])):
            ave[j] += data[i][j]
    for i in range(len(ave)):
        ave[i] /= len(data)
    return ave

def getCovarianceMatrix(data, average):
    cov = np.zeros(shape=(len(data[0]),len(average)), dtype=np.float32)
    for x in data:
        sub = np.array(x-average, dtype=np.float32)[np.newaxis]
        cov += np.dot(sub.T, sub) 
    cov /= len(data)
    return cov

def getVariables(data):
    averages = {}
    covariances = {}
    for cat in data.keys():
        averages[cat] = getAverageVector(data[cat])
        covariances[cat] = getCovarianceMatrix(data[cat], averages[cat])
    return averages, covariances

def getPriorProbabilitiesForCategories(data, totalDataPoints):
    pp = {}
    totalDataPoints = 0.0
    for cat in data.keys():
        totalDataPoints += len(data[cat])
    for cat in data.keys():
        pp[cat] = len(data[cat])/totalDataPoints
    return pp

# calculates probability of X being described by the gaussian distribution defined by variables, using the prior probability of the category
# X should be a data point, variables is the tupel (averageVect, covarianceMat), pp is a number from 0 to 1
def getProbabilityOfCategoryGivenX(X, variables, pp):
    ave, cov = variables
    xMinAve = np.array(X-ave)[np.newaxis].T
    return math.log(pp)+.5*(math.log(np.linalg.det(np.linalg.inv(cov)))-np.dot(xMinAve.T, np.dot(np.linalg.inv(cov), xMinAve)))

def getTotalDataPoints(data):
    totalDataPoints = 0.0
    for cat in data.keys():
        totalDataPoints += len(data[cat])
    return totalDataPoints

def findMaxValueIndex(alist):
    m = None
    maxIndex = -1
    for i, value in enumerate(alist):
        if m is None or value > m:
            maxIndex = i
            m = value
    return maxIndex

def discriminitiveAnalysis(cats, probCatGivenX):
    DAinFavor = [0]*len(cats)
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            DA = probCatGivenX[i]-probCatGivenX[j]
            if DA > 0:
                DAinFavor[i] += 1
            else:
                DAinFavor[j] += 1
    # determine which category had the most DAs in its favor
    return findMaxValueIndex(DAinFavor)

# error data is useful for debugging as it shows which entries were errors
def getErrors(data, variables, totalDataPoints):
    errors = 0
    errorDatas = []
    priorProbabilities = getPriorProbabilitiesForCategories(data, totalDataPoints)
    cats = list(data.keys())
    for category in cats:
        for X in data[category]:
            probCatGivenX = []
            for cat in cats:
                probCatGivenX.append(getProbabilityOfCategoryGivenX(X, (variables[0][cat], variables[1][cat]), priorProbabilities[cat]))
            estimatedCategoryIndex = discriminitiveAnalysis(cats, probCatGivenX)
            if category != cats[estimatedCategoryIndex]:
                # error in estimation
                errors += 1
                errorDatas.append([X, cats[estimatedCategoryIndex], category])
    return errors, errorDatas

# returns a dictionary where every key maps to an average value of the covariance matricies
def averageCovariances(covariances):
    aveCov = np.zeros(covariances[list(covariances.keys())[0]].shape, dtype=np.float32)
    for cat in covariances.keys():
        aveCov += covariances[cat]
    return aveCov/len(covariances.keys())

def makeDiagonalMat(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if i != j:
                mat[i][j] = 0

# returns the error rate and errors points for each data point in data relative to its categories variables
# data should be a dictionary whos keys are the categories
# variables should be the tuple (averagesDict, covariancesDict) where each dict has keys for the categories
# discAnalType should be either 'QDA' or 'LDA'
def getErrorInfoFromData(data, variables, discAnalType):
    totalDataPoints = getTotalDataPoints(data)
    if 'LDA' in discAnalType:
        aveCov = averageCovariances(variables[1])
        for cat in variables[1].keys():
            variables[1][cat] = aveCov
    if 'diag' in discAnalType:
        for cat in variables[1].keys():
            makeDiagonalMat(variables[1][cat])
    errors, errorDatas = getErrors(data, variables, totalDataPoints)
    errorRate = errors/totalDataPoints
    return errorRate, errorDatas

# returns a list of linearly seperable categories
def getLinearlySeperableCategories(errorDatas, data):
    linSepCats = []
    cats = []
    for e in errorDatas:
        if e[2] not in cats:
            cats.append(e[2])
    for cat in data.keys():
        if cat not in cats:
            linSepCats.append(cat)
    return linSepCats

def reportResults(qdaErrorRate, ldaErrorRate, linSepCats):
    print("QDA error rate: " +  str(qdaErrorRate*100) + "%")
    print("LDA error rate: " +  str(ldaErrorRate*100) + "%")
    if len(linSepCats) > 0:
        print("Linear seperable categories:")
        for cat in linSepCats:
            print("\t"+str(cat));
    else:
        print("No categories were linearly seperable")

# returns variables generated in training, as well as training error rate and which categories are linearly seperable
def trainOn(data):
    variables = getVariables(data)
    qdaErrorRate, qdaErrorDatas = getErrorInfoFromData(data, variables, 'QDA')
    ldaErrorRate, ldaErrorDatas = getErrorInfoFromData(data, variables, 'LDA')
    linSepCats = getLinearlySeperableCategories(ldaErrorDatas, data)
    return variables, qdaErrorRate, ldaErrorRate, linSepCats

# tests data with the generated variables, returns error rates and which categories are linearly seperable
def testWith(data, variables):
    qdaErrorRate, qdaErrorDatas = getErrorInfoFromData(data, variables, 'QDA')
    ldaErrorRate, ldaErrorDatas = getErrorInfoFromData(data, variables, 'LDA')
    linSepCats = getLinearlySeperableCategories(ldaErrorDatas, data)
    return qdaErrorRate, ldaErrorRate, linSepCats

# returns a list of variables it deteremined to be not useful for classification
def determineUnimportantVariables(rawdata, categoryColumn, testPercentage, trainErrorRates, testErrorRates):
    unimportant = []
    for i in range(len(rawdata[0])): # length of data point X, number of features for X
        if i != categoryColumn:
            data = np.delete(rawdata, i, 1)
            runningCatColumn = categoryColumn
            if i < categoryColumn:
                runningCatColumn -= 1
            trainData, testData = splitData(data, runningCatColumn, testPercentage)
            variables, qdaErrorRate, ldaErrorRate, linSepCats = trainOn(trainData)
            if (qdaErrorRate, ldaErrorRate) == trainErrorRates:
                qdaErrorRate, ldaErrorRate, linSepCats = testWith(testData, variables)
                if (qdaErrorRate, ldaErrorRate) == testErrorRates:
                    unimportant.append(i)
    return unimportant

# train using a diagonal covariance (sigma) matrix
def trainDiagonallyOn(data):
    variables = getVariables(data)
    qdaErrorRate, qdaErrorDatas = getErrorInfoFromData(data, variables, 'QDA-diag')
    ldaErrorRate, ldaErrorDatas = getErrorInfoFromData(data, variables, 'LDA-diag')
    linSepCats = getLinearlySeperableCategories(ldaErrorDatas, data)
    return variables, qdaErrorRate, ldaErrorRate, linSepCats

# test using a diagonal covariance (sigma) matrix
def testDiagonallyWith(data, variables):
    qdaErrorRate, qdaErrorDatas = getErrorInfoFromData(data, variables, 'QDA-diag')
    ldaErrorRate, ldaErrorDatas = getErrorInfoFromData(data, variables, 'LDA-diag')
    linSepCats = getLinearlySeperableCategories(ldaErrorDatas, data)
    return qdaErrorRate, ldaErrorRate, linSepCats

# runs test on dataset
def processData(url, shuffled, categoryColumn, columnNames, testPercentage):
    rawdata = getData(url, columnNames, shuffled)
    if columnNames is None:
        columnNames = range(len(rawdata[0]))
    trainData, testData = splitData(rawdata, categoryColumn, testPercentage)

    variables, trainQDAErrorRate, trainLDAErrorRate, linSepCats = trainOn(trainData)
    print("Training:")
    reportResults(trainQDAErrorRate, trainLDAErrorRate, linSepCats)

    print("\nTesting:")
    testQDAErrorRate, testLDAErrorRate, linSepCats = testWith(testData, variables)
    reportResults(testQDAErrorRate, testLDAErrorRate, linSepCats)

    print("\nUnimportant variables:")
    unimportantVarIndexes = determineUnimportantVariables(rawdata, categoryColumn, testPercentage, (trainQDAErrorRate, trainLDAErrorRate), (testQDAErrorRate, testLDAErrorRate))
    for i in unimportantVarIndexes:
        print("\t"+str(columnNames[i]))

    print("\nTraining with diagonal covariance matrix:")
    variables, trainQDAErrorRate, trainLDAErrorRate, linSepCats = trainDiagonallyOn(trainData)
    reportResults(trainQDAErrorRate, trainLDAErrorRate, linSepCats)
    
    print("\nTesting with diagonal covariance matrix:")
    testQDAErrorRate, testLDAErrorRate, linSepCats = testDiagonallyWith(testData, variables)
    reportResults(testQDAErrorRate, testLDAErrorRate, linSepCats)

if __name__ == "__main__":
    parser = optparse.OptionParser(usage="./DiscriminitiveAnalysis.py -d dataset_url -c category_column_number -t test_split_percentage")
    parser.add_option('-d', action='store', type='string', dest='dataset', help='The path to the dataset to categoryize', default=None)
    parser.add_option('-s', '--shuffle', action='store_true', help='Shuffle the dataset before splitting', dest='shuffle')
    parser.add_option('-c', '--categoryColumn', action='store', type='int', dest='categoryColumn', help="The column to use as categories", default=None)
    parser.add_option('-t', '--testPercentage', action='store', type='int', dest='testPercentage', help="The percent of the data to allocate for testing data", default=None)
    parser.add_option('--iris', action='store_true', dest='iris', default=False)
    options, args = parser.parse_args()
    url = ""
    columnNames = []
    if options.dataset is not None and options.categoryColumn is not None and options.testPercentage is not None:
        if(options.iris):
            columnNames = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
        columnNames = None
        processData(options.dataset, options.shuffle, options.categoryColumn, columnNames, options.testPercentage)
    else:
        parser.print_help()
