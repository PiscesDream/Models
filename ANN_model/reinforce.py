import ann_rein
from cPickle import load
from numpy import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    train_x, train_y, test_x, test_y = load(open('digit_data.dat', 'rb'))

    ind = random.permutation(test_x.shape[0])
    test_x = array(test_x)[ind]
    test_y = array(test_y)[ind]
    ind = random.permutation(train_x.shape[0])
    train_x = array(train_x)[ind]
    train_y = array(train_y)[ind]
    m = test_x.shape[0]
    m_ = train_x.shape[0]
    
    
    cl = ann_rein.NeuralNetworkClassifier( train_x.shape[1], 10, [10], method_specific_options = {'maxiter':100})
    cl.fit(train_x, train_y)
    correct = (cl.predict(test_x) == test_y).sum()
    print '%003d/%d : %d/%d = %lf' % (-1, m_, correct, m, (correct * 1.0 / m))

    rec = []
    cl = ann_rein.NeuralNetworkClassifier( train_x.shape[1], 10, [10])
    for i in xrange(train_x.shape[0]):
        x = train_x[:i]
        y = train_y[:i]
        cl.fit(x, y)
#        x = train_x[i]
#        y = train_y[i]
#        cl.fit(array([x]), array([y]))
        correct = (cl.predict(test_x) == test_y).sum()
        print '%003d/%d : %d/%d = %lf' % (i, m_, correct, m, (correct * 1.0 / m))
        rec.append(correct)            
    
    plt.plot(arange(len(rec)), rec)
    plt.show()
