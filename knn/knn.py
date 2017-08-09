from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    #inX is the dateSet to be tested
    dataSetSize = dataSet.shape[0]
    #shape[0] is number of rows for dataSetSize
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #tile is a function which will use inX to creat a dataSetSize matrix
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    #axis = 1 means sum each row
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    #sort from small to large numbers
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) +1
        #caculate total number of each labels
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #choose the the label which has the most items(Biggest classCount)
    return sortedClassCount[0][0]

def file2matrix(filename) :
    fr = open(filename)
    arrayOLines = fr.readlines()
    #get line from data file; Each line of array looks like '40920\t8.326976\t0.953952\tlargeDoses\r\n'
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    """>>> zeros((2,3))
       array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])
    """
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        """>>> line
               '40920\t8.326976\t0.953952\tlargeDoses\r\n'
           >>> line = line.strip()
           >>> line
               '40920\t8.326976\t0.953952\tlargeDoses'
        """
        listFromLine = line.split('\t')
        #listFromLine: ['40920', '8.326976', '0.953952', 'largeDoses']
        returnMat[index, :] = listFromLine[0:3]
        """>>> listFromLine[0:3] :['40920', '8.326976', '0.953952'] Get data not label
        >>> returnMat =zeros((1,3))
        >>> returnMat :    array([[ 0.,  0.,  0.]])
        >>> returnMat[0, :] = listFromLine[0:3]
        >>> returnMat : array([[  4.09200000e+04,   8.32697600e+00,   9.53952000e-01]])
        returnMat is the dataSet from the first three collums of files
        """
        classLabelVector.append(listFromLine[-1])
        #Labels are stored
        index +=1
    return returnMat,classLabelVector
