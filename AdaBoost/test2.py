from numpy import *
import adaboost

def loadData(filename):
    x = []
    y = []
    for line in open(filename):
        a = map(lambda x: float(x), line.strip().split('\t'))
        x.append(a[:-1])
        y.append(a[-1])
    return array(x), array(y)

if __name__ == '__main__':
    x, y = loadData('in.txt')
    print '=' * 80
    ada = adaboost.AdaBoost(x,y, 100)
