from numpy import *
import operator

class KNN(object):
    
    def __init__(self, train_x, train_y, k = 3):
        self.train_x = train_x
        self.train_y = train_y
        self.trainSize = train_x.shape[0]
        self.k = k
#        print self.train_x.shape
#        print self.trainSize
        self.train()

    def predict(self, test_x):
 
        self.test_x = test_x
        self.testSize = test_x.shape[0]

        self.predictions = zeros((self.trainSize, 1))

        for index, predictData in enumerate(self.test_x):
            print '%lf\r' % ((index + 1) * 1.0 / self.testSize * 100),

            diffMat = tile(predictData, (self.trainSize, 1)) - self.train_x
            distances = ((diffMat ** 2).sum(axis = 1)) ** 0.5
            sortedDistIndicies = distances.argsort()
            classCount = {}
            for i in xrange(self.k):
                voteIlabel = self.train_y[sortedDistIndicies[i]]
                classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
            sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
            self.predictions[index] = sortedClassCount[0][0]

        return self.predictions
