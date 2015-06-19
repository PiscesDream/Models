from numpy import *
import matplotlib.pyplot as plt

def plt_ion():
    plt.ion()

def plt_show():
    plt.show()

def plot(food, ants, obstacle, pheromone):

    plt.clf()
    
    for ele in food:
        plt.plot(ele[0], ele[1], ms = 10, color = 'blue', marker = 'x')

    for ele in ants:
        plt.plot(ele[0], ele[1], color = 'black', marker = 'o')

    for ele in obstacle:
        plt.plot(ele[0], ele[1], color = 'red', ms = 20, marker = 'o', alpha = .5)

    for ele in pheromone:
        plt.plot(ele[0], ele[1], color = 'green', alpha = ele[2], marker = 'o')

    plt.draw()
        

if __name__ == '__main__':

    plt.ion()
    plt.show()
    for i in xrange(100):
#        plt.clf()
        plot(random.rand(1,2), random.rand(1,2), random.rand(1,2), random.rand(3,3) )
        plt.draw()
