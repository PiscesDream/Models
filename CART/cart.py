from numpy import *

def getMean(T):
    if isTree(T['right']): T['right'] = getMean(T['right'])
    if isTree(T['left']):  T['left']  = getMean(T['left'])
    return (T['left']+T['right'])/2.0

def isTree(obj):
    return type(obj).__name__ == 'dict'

def regLeaf(x):
    return mean(x[:,-1])
def regErr(x):
    return var(x[:, -1]) * x.shape[0]        
def regPredict(model, inDat):
    return float(model)

def linearSolve(x):
    m, n = shape(x)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = x[:, 0:n-1]
    Y = x[:,-1].reshape((m,1))
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot be inversed,\n\
                        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y
def linLeaf(x):
    ws, X, Y = linearSolve(x)
    return ws
def linErr(x): 
    ws, X, Y = linearSolve(x)
    yHat = X * ws
    return sum(power(Y - yHat, 2))
def linPredict(model, x):
    x = mat(x).reshape((1,-1))
    n = x.shape[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = x
    return float(X * model)


def splitx(x, feat, val):
    return x[x[:, feat] < val], x[x[:, feat] >= val]

class CART(object):

    def __str__(self):
        return str(self.retTree)
    
    def __repr__(self):
        return str(self.retTree)

    def __init__(self, x, #y is the last column in x
                leaf_type = regLeaf, err_type = regErr, pre_type = regPredict,
                ops = (1, 4)):
                 
        self.leaf_type = leaf_type
        self.err_type = err_type
        self.pre_type = pre_type
        self.tolS = ops[0]
        self.tolN = ops[1]

        self.retTree = self.createTree(x)

    def createTree(self, x):
        feat, val = self.bestSplit(x)
        if feat == None:
            return val
        retTree = {}
        retTree['spInd'] = feat
        retTree['spVal'] = val
        l, r = splitx(x, feat, val)
        retTree['left'] = self.createTree(l)
        retTree['right'] = self.createTree(r)
        return retTree        
    

    def bestSplit(self, x):
        if len(set(x[:,-1])) == 1:
            return None, self.leaf_type(x)   
        m, n = shape(x)
        Error = self.err_type(x)
        bestError = None
        bestIndex = None
        bestValue = None
        for feat in range(n-1):
            for val in set(x[:, feat]):
                l, r = splitx(x, feat, val)
                if (l.shape[0] < self.tolN) or (r.shape[0] < self.tolN): continue
                tmpErr = self.err_type(l) + self.err_type(r)
                if bestError == None or tmpErr < bestError:
                    bestIndex = feat
                    bestValue = val
                    bestError = tmpErr                    
        if bestError == None or Error - bestError < self.tolS:
            return None, self.leaf_type(x)
        return bestIndex, bestValue        

#------------------------------------predicting-------------------------------------
    def predict(self, x):
        return array(map(lambda x: self.treeForecast(self.retTree, x), x))

    def treeForecast(self, T, x):
        if not isTree(T): return self.pre_type(T, x)
        if x[T['spInd']] < T['spVal']:
            return self.treeForecast(T['left'], x)
        else:
            return self.treeForecast(T['right'], x)        

#------------------------------------post pruning------------------------------------
    def prune(self, x):
        self.retTree = self.prune_sub(self.retTree, x)
          
    def prune_sub(self, T, x):

        if x.shape[0] == 0: return getMean(T)
        if isTree(T['right']) or isTree(T['left']):
            l, r = splitx(x, T['spInd'], T['spVal'])
        if isTree(T['left']): T['left'] = self.prune_sub(T['left'], l)
        if isTree(T['right']): T['right'] = self.prune_sub(T['right'], r)
        if not isTree(T['left']) and not isTree(T['right']):
            l, r = splitx(x, T['spInd'], T['spVal'])
            errorNoMerge = sum(power(l[:,-1] - T['left'], 2)) +\
                           sum(power(r[:,-1] - T['right'], 2))
            treeMean = (T['left'] + T['right'])/2.0
            errorMerge = sum(power(x[:,-1] - treeMean, 2))
            if errorMerge < errorNoMerge:
                #print 'merge'
                return treeMean
            else:
                return T
        else:
            return T          
