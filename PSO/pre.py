from test_fun import *
from numpy import * 
from operator import le, gt
import pso_adv
import pso
import pso_w
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def f(x):
#    return Himmelblau(*x)
    return Rosenbrock(x)

def plot3d(xs, best_pos, best, iteration):
    ax = plt.subplot(111, projection = '3d')

    X = arange(-10, 10, 1)
    Y = arange(-10, 10, 1)
    X, Y = meshgrid(X, Y)
    Z = f([X, Y])

    ax.set_xlabel('X', fontsize = 'large')
    ax.set_ylabel('Y', fontsize = 'large')
    ax.set_zlabel('Z', fontsize = 'large')
    
#    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#        linewidth=0, antialiased=False)
    ax.plot_wireframe(X, Y, Z)
    ax.scatter(xs[0], xs[1], f([xs[0], xs[1]]), color='red')

    plt.draw()
    plt.show()


def plot(xs, best_pos, best, iteration):
    a = linspace(-5.12, 5.12, 100)
    plt.clf()
    plt.plot(a, map(lambda x: f(x), a), color = 'blue')
    ys = map(lambda x: f(x), xs)
    plt.plot(xs, ys, 'ro')    
    plt.draw()

    plt.pause(0.3)

def plot2(xs, best_pos, best, iteration, w = None):
    print '\t',iteration, '\t\tf(',best_pos,') =', best, 'w =', w, '\r',

#    plt.pause(0.3)

if __name__ == '__main__':   
#    plt.plot(a, map(lambda x: Rastrigin([x]), a))    
#    plt.show()


    plt.ion()
    plt.show()

    dimension = 2
    for iteration in xrange(50):
        print 'test %02d:'% iteration

        print('pso_ori')
        x = pso.PSO(f, le, lambda:(random.rand(dimension)-.5)*10,
            50, lambda x: all(x<5) and all(x>5), 10,           
            w = 0.8, c1 = 2, c2 = 2,
            maxiter = 1000,
            plot_f = plot3d).get_ans()

