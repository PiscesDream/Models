from numpy import *

class Standard_Regression(object):
    
    def __init__(self, x, y, withone = False):        
        if withone == False:
            xMat = hstack((ones((x.shape[0], 1)), mat(x) ))
        else:
            xMat = mat(x)            
        yMat = mat(y).T
        xTx = xMat.T * xMat
        if linalg.det(xTx) == 0.0:
            print "This matrix is singular, cannot do inverse"
            return 
        self.ws = linalg.solve(xTx, xMat.T * yMat)

    def getw(self):
        return self.ws

    def predict(self, x, withone = False):
        if withone == False:
            xMat = hstack((ones((x.shape[0], 1)), mat(x) ))
        else:
            xMat = mat(x)
        yHat = xMat * self.ws        
        return yHat

class LWLR(object):
    
    def __init__(self, x, y, k = 1.0, withone = False):
        if withone == False:
            self.x = hstack((ones((x.shape[0], 1)), mat(x) ))
        else:
            self.x = x
        self.y = y
        self.k = k

    def lwlr(self, testPoint):
        xMat = mat(self.x)
        yMat = mat(self.y).T
        m = xMat.shape[0]
        weights = mat(eye(m))
        for j in xrange(m):
            diffMat = testPoint - xMat[j,:]
            weights[j,j] = exp(diffMat * diffMat.T/(-2.0 * self.k ** 2))
        xTx = xMat.T * (weights * xMat)
        if linalg.det(xTx) == 0.0:
            print "This matrix is singular, cannot be inverse"
            return 
        ws = linalg.solve(xTx, xMat.T * (weights * yMat))
        return testPoint * ws

    def predict(self, test_x):            
        m = test_x.shape[0]
        yHat = zeros(m)
        for i in xrange(m):
            yHat[i] = self.lwlr(test_x[i])
        return yHat
