from numpy import *
from scipy import optimize

class AutoEncoder(object):
    
    def __init__(self, hidden_layers = None, lmbd = 1.0,  beta = 1.0, sparsity = 0.05 ,
        opt_method = 'L-BFGS-B', opt_options = None):

        if hidden_layers is None:
            self.hidden_layers = [100]
        else:
            self.hidden_layers = hidden_layers
        self.lmbd = lmbd
        self.beta = beta
        self.sparsity = sparsity
        self.opt_method = opt_method
        self.opt_options = opt_options
        if opt_method in ['L-BFGS-B', 'BFGS', 'CG', 'Newton-CG']:
            if self.opt_options is None:
                self.opt_options = {}
        self.opt_options.setdefault('maxiter', 100)

        self.L = 2 + len(self.hidden_layers)
        self.A = [None] * self.L
        self.Z = [None] * self.L
        self.sparsity_ave = [None] * self.L
        self.theta = [None] * (self.L - 1)
        self.delta = [None] * self.L
        self.grad = [None] * self.L

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + exp(-z))

    @staticmethod
    def _unroll(x):
        return concatenate(map(lambda x: x.flatten(), x))

    @staticmethod
    def _KL(a, b):
        return a * log(a / b) + (1 - a) * log((1-a) / (1-b))

    @staticmethod
    def _KL_grad(a, b):
        return - a/b + (1-a)/(1-b)

    def _f(self, theta):
        self.A[0] = self.X
        m = float(self.X.shape[0])

        theta_idx = 0

        for i in range(self.L - 1):
            theta_shape = (self.layers[i] + 1, self.layers[i+1])
            theta_len = theta_shape[0] * theta_shape[1]
            self.theta[i] = theta[theta_idx : theta_idx + theta_len].reshape(theta_shape)
            theta_idx += theta_len

            self.Z[i+1] = hstack((
                ones((self.A[i].shape[0], 1)),
                self.A[i])).dot(self.theta[i])
            self.A[i+1] = self._sigmoid(self.Z[i+1])

            self.sparsity_ave[i+1] = 1/m * self.A[i+1].sum(axis=0)            

#        mJ = - (self.y * log(self.A[-1]) + (1 - self.y) * log(1 - self.A[-1])).sum() \
        mJ =  0.5/m * ((self.A[-1] - self.y) ** 2).sum() \
             + self.lmbd * 0.5 * (theta ** 2).sum() \
             + self.beta * sum(map(lambda x: self._KL(self.sparsity, x).sum(), self.sparsity_ave[1:-1]))
            
        self.delta[-1] = self.A[-1] - self.y
        for i in range(self.L-2, 0, -1):
            self.delta[i] = (self.delta[i+1].dot(self.theta[i].T[:,1:]) +\
                    self.beta * self._KL_grad(self.sparsity, self.sparsity_ave[i])) * self.A[i] * (1-self.A[i])
    
        for i in range(self.L - 1):
            self.grad[i] = vstack((ones((1, self.A[i].shape[0])),self.A[i].T)).dot(self.delta[i+1])
#            print zeros((1,self.layers[i+1])).shape
#            print self.theta[i][1:,:].shape
            self.grad[i] += self.lmbd * vstack((zeros((1, self.layers[i+1])), self.theta[i][1:,:]))

#        print 'A1',self.A[1][0]
#        print 's1',self.sparsity_ave[1][0]
#        print 'spa_cost',self.sparsity_ave[1]
        print 'spa_cost',self.beta * sum(map(lambda x: self._KL(self.sparsity, x).sum(), self.sparsity_ave[1:-1]))
#        print 'spa_grad',self.beta * self._KL_grad(self.sparsity, self.sparsity_ave[i])
#        print 'grad',self.grad[0]                
#        print 'theta',self.theta[0]

        Grad = self._unroll(self.grad[:-1])

        return (mJ, Grad)

    def fit(self, X, y):
        self.X = X
        self.y = y
        try:
            y.shape[1]
            self.single_y = False
        except:
            target_labels = sorted(list(set(y)))
            labels_count = len(target_labels)
            self.map_labels_index = dict(zip(target_labels, range(labels_count)))
            self.map_index_labels = dict(zip(range(labels_count), target_labels))

            self.y = zeros((X.shape[0], labels_count))
            for i, label in enumerate(y):
                self.y[i, self.map_labels_index[label]] = 1
            self.single_y = True

        self.layers = [X.shape[1]]
        self.layers.extend(self.hidden_layers)       
        self.layers.append(self.y.shape[1])

        init_thetas = [None] * (self.L - 1)
        for i in range(self.L - 1):
            epsilon = sqrt(6.0 / (self.layers[i] + self.layers[i+1]))
            init_thetas[i] = random.mtrand.rand(self.layers[i] +1,
                    self.layers[i+1]) * 2.0 * epsilon - epsilon
        
        init_theta = self._unroll(init_thetas)
        self.init_theta = init_theta

        self.result = optimize.minimize(self._f, x0 = init_theta,
            method = self.opt_method, jac = True,
            options = self.opt_options)

        self.optimized_theta = []
        optimized_theta = self.result.x
        theta_idx = 0
        for i in range(self.L - 1):
            theta_shape = (self.layers[i] + 1, self.layers[i+1])
            theta_len = theta_shape[0] * theta_shape[1]
            self.optimized_theta.append(
                optimized_theta[theta_idx:theta_idx+theta_len].
                reshape(theta_shape))
            theta_idx += theta_len

    def predict(self, X):
        self.A[0] = X
        m = X.shape[0]

        for i in range(self.L - 1):
            self.Z[i+1] = hstack((ones((m, 1)), self.A[i]))
            self.A[i+1] = self._sigmoid(self.Z[i+1].dot(self.optimized_theta[i]))
        if self.single_y:
            return map(lambda x: self.map_index_labels[x], self.A[-1].argmax(axis = 1))
        else:
            return self.A[-1]
