from copy import deepcopy
import matplotlib.pyplot as plt
from time import sleep
from numpy import *
import ga
import ga2

DataSize = 0
data = 0 
factorial_table = []

def createFactorialTable(length):
    global factorial_table
    base = 1
    for i in range(length):
        factorial_table.append(base)
        if i != 0:
            base *= i

def tsp_fitness_f(route):

    global data
    
    def p2p(p1, p2):
        return ((p1[0]-p2[0]) ** 2+(p1[1]-p2[1]) ** 2) ** 0.5

    length = .0
    for index, point in enumerate(route):
        if index == len(route)-1:
            length += p2p(data[point], data[route[0]])
        else:
            length += p2p(data[point], data[route[index+1]])

    return length    

def tsp_gen_init():
    global DataSize
    temp = random.permutation(DataSize)
    return temp                

def tsp_mutation_hillclimb(base):
    ori_fitness = tsp_fitness_f(base)
    while True:
#        posx = random.randint(0,len(base))
#        posy = random.randint(0,len(base))
#        t = copy(base[posx])
#        base[posx] = copy(base[posy])
#        base[posy] = copy(t)
        base = tsp_gen_init()

        if tsp_fitness_f(base)>ori_fitness:
            break;
    return base



def tsp_mutation_f(base):
#    if random.rand()<0.2: return tsp_gen_init()
    for i in xrange(random.randint(0,len(base))):
        posx = random.randint(0,len(base))
        posy = random.randint(0,len(base))
#        t = copy(base[posx])
#        base[posx] = copy(base[posy])
#        base[posy] = copy(t)

        base[posx], base[posy] = base[posy], base[posx]

    return base               

def tsp_plot_f(route, clr = 'red', clean = False):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plt.plot(route[:,0]+route[0,0], route[:,1]+route[-1,0], color='red', marker='x', ms=10 )
    
    px = []
    py = []
    for point in route:
        px.append(data[point ,0])
        py.append(data[point ,1])
    px.append(data[route[0],0])
    py.append(data[route[0],1]);

    if clean:
        plt.clf()
    plt.plot(px, py , color=clr, marker='x', ms=10 )
    plt.draw()
#    sleep(0.1)

#============================================================================
#============================================================================
#============================================================================
#============================================================================
#============================================================================
    
#ref = http://blog.csdn.net/xuyuanfan/article/details/6726477
def tsp_crossover_sgc(father, mother):
    '''singal cross'''
    length = len(father)
    cp = random.randint(0, length)
    child1 = list(father[:cp])
    child2 = list(mother[:cp])
    for ele in mother:
        if not ele in child1:
            child1.append(ele)
    for ele in father:
        if not ele in child2:
            child2.append(ele)
    return child1, child2

def tsp_crossover_pmx(father, mother):
    '''pmx'''
    length = len(father)
    c1 = random.randint(0, length)
    c2 = random.randint(0, length)
    if c1 > c2:
        c1, c2 = c2, c1
    block1 = father[c1:c2+1]
    block2 = mother[c1:c2+1]
    d1 = dict(zip(block1, block2))
    d2 = dict(zip(block2, block1))
    child1 = deepcopy(father)
    child1[c1:c2+1] = block2
    child2 = deepcopy(mother)
    child2[c1:c2+1] = block1
    for i in xrange(length):
        if i < c1 or i > c2:
            if child1[i] in d2:
                child1[i] = d2[child1[i]]
            if child2[i] in d1:
                child2[i] = d1[child2[i]]

#    print '-' * 80
#    print father, mother
#    print c1, c2
#    print child1, child2
#    raw_input()

    return child1, child2


def tsp_crossover_HGA(father, mother):
    def swirl(route, pos):
        route.append(route[pos])
        route.pop(pos)

    def dist(p1, p2):
        def p2p(p1, p2):
            return ((p1[0]-p2[0]) ** 2+(p1[1]-p2[1]) ** 2) ** 0.5
        global data
        return p2p(data[p1], data[p2])

    father = list(copy(father))
    mother = list(copy(mother))

    child = [0]
    ind = 0
    while ind < len(father)-1:
        while (father[ind] != child[ind]):
            swirl(father, ind)
        while (mother[ind] != child[ind]):
            swirl(mother, ind)
        if (dist(child[ind], father[ind+1]) < dist(child[ind], mother[ind+1])):
            child.append(father[ind+1])
        
        else:
            child.append(mother[ind+1])
        ind += 1
    return child

   
def tsp_crossover_mixed(father, mather):
    if random.rand()< 0.8:
        return tsp_crossover_HGA(father, mather)
    else:
        return tsp_crossover_sgc(father, mather)

#============================================================================
#============================================================================
#============================================================================
#============================================================================


if __name__ == '__main__':
    global DataSize, data
    data = load('data.npy')
    DataSize = data.shape[0]
    createFactorialTable(DataSize)

    
#    def __init__(self, fitness_f, terminator, generation_size, genertaion_init, 
#        crossover_rate, crossover_f, mutation_rate, mutation_f, plot = False, plot_f = None):

#    fig = plt.figure()
    plt.ion()
    plt.show()
    plt.plot(1,1)
    raw_input('sizing')
    
    '''    MyGA = ga.GA(tsp_fitness_f, {'fitness_thresold':0}, 300, tsp_gen_init,
        0.9, tsp_crossover_HGA, 0.3, tsp_mutation_f, True, tsp_plot_f, 10, lambda a,b: a<b)
    MyGA.fit()'''


#    def __init__(self, fitness_f, terminator, generation_size, genertaion_init, 
#        crossover_vs_survival, crossover_f, mutation_rate, mutation_f, plot = False, plot_f = None):
    MyGA = ga2.GA(tsp_fitness_f, {'fitness_thresold':0}, 300, tsp_gen_init,
                        0.95, tsp_crossover_HGA, 0.05, tsp_mutation_f, True, tsp_plot_f, 2, lambda a,b: a<b)
    MyGA.fit()
