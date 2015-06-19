import ann
import pso_ann
from numpy import *
from cPickle import dump
import os

def  img2vector(filename):  
    rows = 32  
    cols = 32  
    imgVector = zeros((1, rows * cols))   
    fileIn = open(filename)  
    for row in xrange(rows):  
        lineStr = fileIn.readline()  
        for col in xrange(cols):  
            imgVector[0, row * 32 + col] = int(lineStr[col])  
  
    return imgVector  

def loadDataSet():  
    ## step 1: Getting training set  
    print "---Getting training set..."  
    dataSetDir = os.getcwd()+'/digits/'
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits') # load the training set  
    numSamples = len(trainingFileList)  
  
    train_x = zeros((numSamples, 1024))  
    train_y = []  
    for i in xrange(numSamples):  
        filename = trainingFileList[i]  
  
        # get train_x  
        train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename)   
  
        # get label from file name such as "1_18.txt"  
        label = int(filename.split('_')[0]) # return 1  
        train_y.append(label)  
  
    ## step 2: Getting testing set  
    print "---Getting testing set..."  
    testingFileList = os.listdir(dataSetDir + 'testDigits') # load the testing set  
    numSamples = len(testingFileList)  
    test_x = zeros((numSamples, 1024))  
    test_y = []  
    for i in xrange(numSamples):  
        filename = testingFileList[i]  
  
        # get train_x  
        test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename)   
  
        # get label from file name such as "1_18.txt"  
        label = int(filename.split('_')[0]) # return 1  
        test_y.append(label)  
  
    return train_x, train_y, test_x, test_y  
  
# test hand writing class  
def testHandWritingClass():  
    ## step 1: load data  
    print "step 1: load data..."  
    train_x, train_y, test_x, test_y = loadDataSet()  

    dump( (train_x, train_y, test_x, test_y), open('digit_data.dat', 'wb'),True)


#    print train_x[0].shape
#    print test_x[0].shape
  
    ## step 2: training...  
    print "step 2: training..."  
#=========================BP ANN======================================
    Cl = ann.NeuralNetworkClassifier(hidden_layers = [10] ,
        method_specific_options = {'maxiter':100,'disp':True});
#========================PSO ANN======================================
#    Cl = pso_ann.PSO_ANN([10], p_size = 20, maxiter = 50, w = 1.2, c1 = 2, c2 = 2, vmax = 6)
# 
#=====================================================================

    Cl.fit(train_x, train_y)
#    print Cl.optimized_theta

    ## step 3: testing  
    print "step 3: testing..."  
    numTestSamples = test_x.shape[0]  
    matchCount = 0  
    predict = Cl.predict(test_x)
    for i in xrange(numTestSamples):  
        if predict[i] == test_y[i]:  
            matchCount += 1  
    accuracy = float(matchCount) / numTestSamples  
  
    ## step 4: show the result  
    print "step 4: show the result..."  
    print 'The classify accuracy is: %.2f%%' % (accuracy * 100)  

if __name__ == '__main__':
    testHandWritingClass();
