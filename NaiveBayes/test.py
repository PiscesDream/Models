import re
from math import log, exp
import nb
from numpy.random import randint

def textPrase(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


test_num = 40

if __name__ == '__main__':

    x = []
    y = []

    for i in range(1, 26):
        x.append( textPrase(open('email/spam/%d.txt' % i).read()) )
        y.append('spam')

        x.append( textPrase(open('email/ham/%d.txt' % i).read()) )
        y.append('ham')

    test_x = []
    test_y = []
    for i in xrange(test_num):
        rand = randint(0, len(y))
        test_x.append(x.pop(rand))
        test_y.append(y.pop(rand))

#    classifier = nb.NaiveBayes(x, y, lambda x: log(x), lambda x,y: x+y, lambda x: exp(x))
#    classifier = nb.NaiveBayes(x, y, lambda x: x*1000, lambda x,y: x*y, lambda x: x/1000)
    classifier = nb.NaiveBayes(x, y)
    correct = 0
    for test, ans in zip(test_x, test_y):
        result = classifier.classify(test, True)
        print result[2], result[0], ans
        if result[0] == ans:            
            correct += 1
    print 'correct rate = %lf' % (correct*1.0/test_num)
