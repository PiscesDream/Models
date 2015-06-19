import log_reg 
from numpy import *

testNum = 10

if __name__ == '__main__':

    data = []
    for line in open('horseColicTraining.txt'):
        data.append(array(map(lambda x: float(x), line.strip().split('\t'))))
    test_data = []
    for line in open('horseColicTest.txt'):
        test_data.append(array(map(lambda x: float(x), line.strip().split('\t'))))
        
    data = array(data)
    test_data = array(test_data)

    x = data[:,:-1]
    y = data[:,-1]    
    test_x = test_data[:,:-1]
    test_y = test_data[:,-1]

    classifier = log_reg.Logistic_Regression(x, y, 1.0)
#    print test_x[0]
    print '%d/%d' % ((classifier.predict(test_x) == test_y).sum(), test_y.shape[0])
