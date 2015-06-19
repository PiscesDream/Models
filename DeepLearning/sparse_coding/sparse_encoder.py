from numpy import *
import read_and_plot as rap
from scipy.optimize import minimize

def test_grad(f, theta, cmpor, num = None):
    '''some bug in it'''
    EPSILON = 1e-6
    if num == None: num = theta.shape[0]
    grad = zeros(num)
    print 'total theta', num
    for ind in xrange(num):
        add = theta.copy()        
        add[ind] += EPSILON
        sub = theta.copy()
        sub[ind] -= EPSILON

        grad[ind] = (f(add) - f(sub))/(2 * EPSILON)
        print '\rchecking theta', ind

    print hstack((grad.reshape(-1,1), (cmpor[:num]).reshape(-1,1)))

def disp(a, b, c, d):
    print 'W1:', a
    print 'W2:', b
    print 'b1:', c
    print 'b2:', d

def p(s, a):
    print s, a

class sparseEncoder(object):
    ''' with 3 layers'''        
    @staticmethod
    def sigmoid(z):
        return 1/(1+exp(-z))

    def __init__(self, visible_size, hidden_size = 16,
                     lmbd = 0.0001, sparse_para = 0.01, beta = 3, ops = {'maxiter':400, 'disp':True}):

        self.ops = ops
        self.hs = hidden_size
        self.lmbd = lmbd
        self.sp = sparse_para
        self.beta = beta
        
        #====================init theta==============================
        n = visible_size
        r  = sqrt(6) / sqrt(n+hidden_size+1)
        W1 = random.rand(n, hidden_size) * 2 * r - r
        W2 = random.rand(hidden_size, n) * 2 * r - r

        b1 = zeros((hidden_size, 1))
        b2 = zeros((n, 1))

#        disp(W1, W2, b1, b2)
        self.theta = concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))
    

    def fit(self, x):
        self.x = x
        self.result = minimize(self.costf, x0 = self.theta,
                        method = 'L-BFGS-B', jac = True,
                        options = self.ops, tol = 1e-100)
        self.theta = self.result.x
   
    def costf(self, theta):
        #=====================roll up=================================
#        p('theta', theta)        

        n1 = self.x.shape[1]
        n2 = self.hs
#        print 'n1, n2 = ', n1, n2
        t = n1 * n2
        
        W1 = theta[: t].reshape(n1, n2)
#        print 'W1 =', W1
        W2 = theta[t: 2 * t].reshape(n2, n1)
#        print 'W2 =', W2
        b1 = theta[2 * t: 2 * t + n2]
#        print 'b1 = ', b1.T
        b2 = theta[2 * t + n2:]        
#        print 'b2 = ', b2.T
#        disp(W1, W2, b1, b2)

        #=====================cost f======================================
#        cost = 0;
        W1grad = zeros(size(W1))
        W2grad = zeros(size(W2))
        b1grad = zeros(size(b1))
        b2grad = zeros(size(b2))

        m, ndims = self.x.shape

#        print 'data =', self.x
        a1 = self.x
        z2 = dot(a1, W1) + b1
#        print 'z2 = ', z2
        a2 = self.sigmoid(z2)
#        print 'a2 = ', a2
        z3 = dot(a2, W2) + b2
#        print 'z3 = ', z3
        a3 = self.sigmoid(z3)        
#        print 'a3 = ', a3

        muls = sum(a2, 0) * 1.0 / m
#        print 'muls =', muls
        para = self.sp
#error here?
        kl = sum( para      * log( para     /   muls) +\
                  (1-para)  * log( (1-para) /   (1-muls) ) )
#        print 'kl =', kl
        cost = .5/m * sum( (a3-self.x)**2 ) +\
               self.lmbd * .5 * ( sum(W1 ** 2) + sum(W2 ** 2) ) +\
               self.beta * kl
#        print 'cost =', cost               

        
        delta3 = -(self.x - a3) * a3 * (1-a3)
#        print 'delta3', delta3
        delta_kl = self.beta * (-para/muls + (1-para)/(1-muls))
#        print 'delta_kl', delta_kl        
        delta2 = (dot(delta3, W2.T) + delta_kl)  * a2 * (1-a2)
#        print 'delta2', delta2

        W2grad = dot(a2.T, delta3)/m + self.lmbd * W2
#        print 'W2grad', W2grad
        b2grad = sum(delta3, 0)/m
#        print 'b2grad', b2grad
        W1grad = dot(a1.T, delta2)/m + self.lmbd * W1
#        print 'W1grad', W1grad
        b1grad = sum(delta2, 0)/m
#        print 'b1grad', b1grad

        grad = concatenate((W1grad.flatten(), W2grad.flatten(), b1grad, b2grad))        
                             #forget the 'grad' in the 1st time, waste an hour and half
                             #mistake '1' and '2' waste 20 mins
        return (cost, grad)


    def visualize_hidden_layer(self):
        '''error?'''
        n1 = self.x.shape[1]
        n2 = self.hs        
        W1 = self.theta[: n1 * n2].reshape(n1, n2)

        s = sqrt(sum(W1 ** 2))
        features = zeros((n2, n1))
        for i in xrange(n2):
            features[i, :] = (W1[:, i]/s).T
        
        rap.all_plot(features)


if __name__ == '__main__':
    test_grad( lambda x: x[0] ** 3 - 2 * x[1], array([1., 2.]), array([3., -2.]) )
    x = rap.load_data(num = 20000, dim = 10)
    cl = sparseEncoder(x.shape[1])
    cl.fit(x)     
    cl.visualize_hidden_layer()

#====================gradient check with octave is neglected=================
#    cl = sparseEncoder(4, 10)
#    cl.x = ones((2, 4))
#    cl.theta = ones((94))
#    cost, grad = cl.costf(cl.theta)   
#    test_grad( lambda x: cl.costf(x)[0], cl.theta, grad, 6)
