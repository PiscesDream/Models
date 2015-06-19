'''
    updated in 08/24/14 in kaggle/Titanic
'''
from copy import copy
from numpy import log2

class ID3(object):
    
    def __init__(self, names, x, y):
        self.names = names
        self.x = x
        self.y = y
        
        self.tree = self.createTree(self.names, self.x, self.y)
#        return self.tree
    def result(self):
        return self.tree

    @staticmethod
    def entropy(S):
        ans = .0
        total = len(S)*1.0
        for ele in set(S):
            p = S.count(ele)/total
            ans += -p*log2(p)
        return ans    

    def choiceBestFeature(self, names, x, y):
        entropy_S = self.entropy(y)
        size_S    = len(y)*1.0
        bestGain = 0
        bestFeatInd = None
        
        for ind in xrange(len(x[0])):
#            print x[0]
#            print names
#            print ind
            featureName = names[ind]
            featureCol  = [row[ind] for row in x]
            featureSet  = set(featureCol)
            S_  =  dict(zip(featureSet, map(lambda x: [], [[]] * len(featureSet)) ))

            for ind2, ele in enumerate(featureCol):
                S_[ele].append(y[ind2])
#            print S_

            Gain = entropy_S
            for ele in S_:
                Gain -= len(S_[ele])/size_S * self.entropy(S_[ele])
#                print ele, self.entropy(S_[ele])
#            print featureName,Gain
            if Gain > bestGain:
                bestGain = Gain
                bestFeatInd = ind
#                print featureName
#                print bestFeatInd
                
        return names[bestFeatInd], bestFeatInd

    def createTree(self, names, x, y):

        if y.count(y[0]) == len(y):
            return y[0]

        if x == []:
            bestClass = None
            bestClassCount = 0
            for ele in set(y):
                temp = y.count(ele)
                if temp > bestClassCount:
                    bestClassCount = temp
                    bestClass = ele
            return bestClass

        bestFeatName, bestFeatInd = self.choiceBestFeature(names, x, y)
        featCol = map(lambda row: row.pop(bestFeatInd), x)       
        names = copy(names)
        names.pop(bestFeatInd)
        splited = dict(zip(set(featCol), map(lambda x: [[],[]],  [[None]] * len(set(featCol)))))
#        print splited
#        print featCol
        for ind, ele in enumerate(x):
            splited[featCol[ind]][0].append(ele)
            splited[featCol[ind]][1].append(y[ind])

#        print 'split with',bestFeatName
#        print names, splited
        for ele in splited:
#            print 'spliting',ele
            splited[ele] = self.createTree(names, splited[ele][0], splited[ele][1])
        return {bestFeatName:splited}
