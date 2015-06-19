from numpy import *
import scipy.optimize as opt

class Logistic_Regression(object):

    def __init__(self, x, y, lamb):
        self.x = hstack(( ones((x.shape[0], 1)), x ))
#        print self.x
        self.y = y
#        print self.y
        self.lamb = lamb

        self.theta = zeros((self.x.shape[1]))
        self.result = opt.minimize(self.cost_grad, jac = True, x0 = self.theta, method = 'L-BFGS-B', options={'disp':False, 'maxiter':400})
        self.theta = self.result.x

    @staticmethod
    def sigmoid(z):
       return 1/(1+exp(-z))

    def cost_grad(self, theta):
        m = self.y.shape[0] * 1.0
        pre = self.sigmoid(dot(self.x, theta))

        thetaTmp = theta.copy()
        thetaTmp[0] = 0
        J = -1/m * (self.y * pre + (1-self.y) * log(1-pre)).sum() + self.lamb / (2*m) * (thetaTmp ** 2).sum()

        error = pre - self.y
        grad = 1/m * dot(error, self.x) + self.lamb/m * theta

#        print '-'*80
#        print J
#        print grad
#        print theta

        return (J, grad)
        
    def result(self):
        return self.theta

    def predict(self, x):
        x = hstack(( ones((x.shape[0], 1)), x))
        return self.sigmoid( dot(x , self.theta) ) >= 0.5 
