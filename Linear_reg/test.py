import line_reg
from numpy import *
import matplotlib.pyplot as plt

def loadData(filename):
    x = []
    y = []
    for line in open(filename):
        a = map(lambda x: float(x), line.strip().split('\t'))
        x.append(a[:-1])
        y.append(a[-1])
    return array(x), array(y)

if __name__ == '__main__':
#    plt.ion()

    x, y = loadData('ex0.txt')
    srtInd = x[:,1].argsort()
    x = x[srtInd]
    y = y[srtInd]

    plt.plot(x[:,1], y, color = 'green', marker = 'o', ls = ' ')

    reger = line_reg.Standard_Regression(x, y, True)
    yHat = reger.predict(x, True)
    print 'correlate:', corrcoef(y, yHat.T)

    plt.plot(x[:,1], yHat, color = 'red', ls = '-')    
    plt.show()



#   lwlr
    plt.plot(x[:,1], y, color = 'green', marker = 'o', ls = ' ')
    
    reger = line_reg.LWLR(x, y, 0.03, True)
    yHat = reger.predict(x)
    print 'correlate:', corrcoef(y, yHat.T)

    plt.plot(x[:,1], yHat, color = 'red', ls = '-')    
    plt.show() 
