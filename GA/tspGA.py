import matplotlib.pyplot as plt
from time import sleep
from numpy import *
import ga

DataSize = 0

def tsp_fitness_f(route):
    
    def p2p(p1, p2):
        return ((p1[0]-p2[0]) ** 2+(p1[1]-p2[1]) ** 2) ** 0.5

    length = .0
    for index, point in enumerate(route):
        if index == route.shape[0]-1:
            length += p2p(point, route[0])
        else:
            length += p2p(point, route[index+1])

    return length    

def tsp_gen_init():
    global data
    temp = copy(data)
    random.shuffle(temp)
    return temp                

def tsp_crossover_f1(father, mother):
    return father
#    child = copy(father)
#    for ele in xrange(len(child)):
#        if (random.rand() > 0.5):
#            child[ele] = mother[ele]

def tsp_mutation_f(base):
    for i in xrange(random.randint(0,len(base))):
        posx = random.randint(0,len(base))
        posy = random.randint(0,len(base))
        t = copy(base[posx])
        base[posx] = copy(base[posy])
        base[posy] = copy(t)
    return base               

#fig = None    

def tsp_plot_f(route):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plt.plot(route[:,0]+route[0,0], route[:,1]+route[-1,0], color='red', marker='x', ms=10 )
    plt.clf()
    plt.plot(route[:,0], route[:,1], color='red', marker='x', ms=10 )
    plt.plot([route[0,0],route[-1,0]], [route[0,1],route[-1,1]], color='red', marker='x', ms=10 )
    plt.draw()
#    sleep(0.1)

if __name__ == '__main__':
    data = load('data.npy')
    global DataSize
    DataSize = data.shape[0]
    
#    def __init__(self, fitness_f, terminator, generation_size, genertaion_init, 
#        crossover_rate, crossover_f, mutation_rate, mutation_f, plot = False, plot_f = None):

#    fig = plt.figure()
    plt.ion()
    plt.show()

    MyGA = ga.GA(tsp_fitness_f, {'iter_maximum':200}, 20, tsp_gen_init,
        .0, tsp_crossover_f1, 1.0, tsp_mutation_f, True, tsp_plot_f)
    MyGA.fit()
