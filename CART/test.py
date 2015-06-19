import cart
from numpy import *
import matplotlib.pyplot as plt
from pprint import pprint

def loadf(filename):
    x = []
    for line in open(filename, 'r'):
        x.append(map(float, line.strip().split('\t')))
    x = array(x)
    return x

if __name__ == '__main__':
    if 0:
        x = loadf('ex0.txt')
        plt.plot(x[:,1],x[:,2],'bx')
        plt.show()
        
        cl = cart.CART(x, ops = (0,1))
        pprint(cl.retTree)

        print '='* 100    
        x2 = loadf('ex2test.txt')
        
        cl.prune(x2)
        pprint(cl.retTree)

    if False:
        x = loadf('sine.txt')
        cl = cart.CART(x, ops = (1, 10))
        yHat = cl.predict(map(lambda x: [x], x[:,0]))

        plt.plot(x[:,0], x[:,1], 'rx')        
        plt.plot(x[:,0], yHat, 'bo')
        plt.show()

        print 'reg:\t', corrcoef(x[:,1], yHat, rowvar = 0)[0, 1]


        cl = cart.CART(x, cart.linLeaf, cart.linErr, cart.linPredict, ops = (1, 10))
        yHat = cl.predict(map(lambda x: [x], x[:,0]))

        plt.plot(x[:,0], x[:,1], 'rx')                
        ind = x[:,0].argsort()
        plt.plot(x[:,0][ind], yHat[ind], 'b')
        plt.show()

        print 'lin:\t', corrcoef(x[:,1], yHat, rowvar = 0)[0, 1]

    if  1:
        train_x = loadf('bikeSpeedVsIq_train.txt')
        test_x = loadf('bikeSpeedVsIq_test.txt')


        reg = cart.CART(train_x, ops = (1, 10))
        reg_yHat = reg.predict(map(lambda x: [x], test_x[:,0]))

        lin = cart.CART(train_x, cart.linLeaf, cart.linErr, cart.linPredict, ops = (1, 20))
        lin_yHat = lin.predict(map(lambda x: [x], test_x[:,0]))
        
        plt.plot(train_x[:,0], train_x[:,1], 'rx')
        plt.show()

        plt.plot(test_x[:,0], test_x[:,1], 'rx')
        plt.plot(test_x[:,0], reg_yHat, 'bo')
        plt.plot(test_x[:,0], lin_yHat, 'go')
        plt.show()

        print 'reg:\t', corrcoef(test_x[:,1], reg_yHat, rowvar = 0)[0, 1]
        print 'lin:\t', corrcoef(test_x[:,1], lin_yHat, rowvar = 0)[0, 1]
