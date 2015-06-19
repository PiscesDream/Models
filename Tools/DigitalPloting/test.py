from pickle import load
import ann
from numpy import *
from plot import all_plot

if __name__=='__main__':
    train_x, train_y, test_x, test_y = load(open('digital.dat','rb'))

    print'training...'
    ann_cl = ann.NeuralNetworkClassifier([100])
    ann_cl.fit(train_x, train_x)

    print 'training done.'
#    pre = ann_cl.predict(test_x)
#    correct = (ann_cl.predict(test_x)==array(test_y)).sum()
#    print 'correct: %d/%d=%lf' %(correct,len(test_y),correct*1.0/len(test_y))

    printing = []
    y = []
    ind = 0
    for j in xrange(16):
        ind = random.randint(0, len(test_y))        
        for i in xrange(1):
            printing.append(ann_cl.A[i+1][ind])
            y.append(pre[ind])
    all_plot(printing, y, 4, 4)
