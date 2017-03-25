from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)

def learnWithRestarts():
    print "Question 5: Learning With Restarts"
    print "\n"
    penDataList = []
    for i in range(5):
        nnet, acc = testPenData();
        penDataList.append(acc)

    print "testPenData:\n"
    print "max:", max(penDataList)
    print "average:", average(penDataList)
    print "standard deviation:", stDeviation(penDataList)
    print "\n\n"

    carDataList = []
    for i in range(5):
        nnet, acc = testCarData()
        carDataList.append(acc)

    print "testCarData:\n"
    print "max:", max(carDataList)
    print "average:", average(carDataList)
    print "standard deviation:", stDeviation(carDataList)
    print "\n\n\n\n"


def varyHiddenLayer():

    print "Question 6: Varying The Hidden Layer"

    print "testPenData:\n"
    penDataDic = {}
    for pc in range(0, 41, 5):
        penDataDic[pc] = []
        for i in range(0, 5):
            nnet, err = testPenData([pc])
            penDataDic[pc].append(err)
        print penDataDic

    for pc in range(0, 41, 5):
        val = penDataDic[pc]
        print pc
        print "max:", max(val)
        print "average:", average(val)
        print "standard deviation:", stDeviation(val)
    print "\n\n"

    print "testCarData:\n"
    carDataDic = {}
    for pc in range(0, 41, 5):
        carDataDic[pc] = []
        for i in range(0, 5):
            nnet, err = testCarData([pc])
            carDataDic[pc].append(err)
        print carDataDic

    for pc in range(0, 41, 5):
        val = carDataDic[pc]
        print pc
        print "max:", max(val)
        print "average:", average(val)
        print "standard deviation:", stDeviation(val)

def XOR():
    examples = ([([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])], [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])])
    iteration = float("inf")
    count = 0
    while iteration != 1:
        if count == 0:
            out = buildNeuralNet(examples, maxItr=100000, hiddenLayerList=[])
        else:
            out = buildNeuralNet(examples, maxItr=100000, hiddenLayerList=[count])
        print '# nodes in the hidden layer: %d \nAge Accuracy: %f accuracy\n\n' % (count, out[1])
        count += 1
        iteration = out[1]
# learnWithRestarts()
# varyHiddenLayer()
XOR()
