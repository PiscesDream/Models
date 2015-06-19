from numpy import *
from operator import gt, le
from pprint import pprint

class AdaBoost(object):

    def __init__(self, x, y, adaNum = 30):
        self.x = x
        self.y = y
        m = y.shape[0]
        self.adaNum = adaNum
        
        D = array([1.0/m] * m)
        DSs = []
        aggPrediction = zeros((m))
        for iterations in xrange(adaNum):
            best_DS, error, prediction = Decision_stump(x, y, D).result()
#            print 'weightD:\t', D            
#            print 'classEst:\t', prediction

            alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))
            best_DS['alpha'] = alpha
            DSs.append(best_DS)

            expon = -1 * alpha * y * prediction
            D = D * exp(expon)
            D = D / D.sum()

            aggPrediction += alpha * prediction 
            aggRate = ( sign(aggPrediction) != y ).sum()*1.0/m

#            print 'aggPre:\t',aggPrediction
            
            if aggRate == 0:
                print 'all fit and exit'
                break
            print 'traning %dth decision stump\r' % (iterations+1)
#            print D
#            print best_DS
#        pprint(DSs)
        self.DSs = DSs
        self.m = m

    def predict(self, x):
        m = x.shape[0]
        aggPrediction = zeros(m)
        for ds in self.DSs:
            argEst = ones(m)
            #argEst[ds['cmpor'](argEst, ds['threshold'])] = 1
            argEst[ds['cmpor'](ds['threshold'], x[:,ds['dim']])] = -1
            aggPrediction += ds['alpha'] * argEst
        return sign(aggPrediction)

class Decision_stump(object):
    
    def __init__(self, x, y, D = None):
        self.DS = {}
        m, n = x.shape
        if D == None:
            D = array([1.0/m] * m)

#        print '-'*80
#        print 'x:\t',x
#        print 'y:\t',y
#        print 'D:\t',D

        minError = inf
        for dim in xrange(n):
            tryPoints = self.createMidPoint(x[:,dim]) 
            for threshold in tryPoints:
                for cmpor in [gt, le]:
                    argEst = zeros((m))
                    argEst[cmpor(x[:,dim], threshold)] = 1
                    argEst[cmpor(threshold, x[:,dim])] = -1
#                    print 'argEst:\t', argEst
                    
                    estErr = (argEst != y) 
#                    print estErr
                    Error = dot(D, estErr)
#                    print Error
                    if Error < minError:
                        minError = Error
                        self.arrEst = argEst.copy()
                        self.minError = minError
                        self.DS['dim'] = dim
                        self.DS['threshold'] = threshold
                        self.DS['cmpor'] = cmpor
#        print minError        
#        print 'Ds:\t',self.DS
#        print 'Pre:\t',self.arrEst

    def result(self):   
        return self.DS, self.minError, self.arrEst

    @staticmethod
    def createMidPoint(l):
#        print l
        l = sorted(list(set(l)))
        p = []
        for ind, ele in enumerate(l):
            try:
                p.append((ele + l[ind+1])/2)
            except:
                pass
        p.append(l[-1] + 0.1)
        p.append(l[0] - 0.1)

#        print p
        return array(p)        
