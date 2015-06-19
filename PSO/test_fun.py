from math import *

def Himmelblau(x, y):
    '''maximum : f(-0.270845, -0.923039) = 181.617
       minimum : f(3.0, 2.0)
                =f(-2.805118, 3.131312)
                =f(-3.779310, -3.283186)
                =f(3.584428, -1.848126) = 0.0'''
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2    

def Rosenbrock(xs, randpara = None):
    '''f(x, y) = (1 - x)**2 + 100 * (y - x**2)**2
        minimum at (1,1,1,...,1)
        randpara belong to (0, 1)'''

    if randpara == None:
        randpara = [1.0] * len(xs)

    f = .0
    for i in xrange(len(xs)-1):
        f += (1 - xs[i])**2 + 100 * randpara[i] * (xs[i+1] - xs[i]**2)**2
    return f

def Rastrigin(xs, A = 10.0):
    '''when A = 10.0, xs belong to [-5.12, 5.12], it has a global minimum at x = 0 where f(x) = 0'''
    return A * len(xs) + sum(map(lambda x: x**2 - A * cos(2 * pi * x), xs))    
