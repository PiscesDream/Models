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
    x, y = loadData('horseColicTraining2.txt')
    
    test_x,test_y = loadData('horseColicTest2.txt')

    ada = adaboost.AdaBoost(x, y, 10)
    dif = (ada.predict(test_x) == test_y).sum()
    print 'correct: %d/%d' % (dif, test_y.shape[0])
    print 'error rate: %lf' % (1-dif*1.0/test_y.shape[0])

