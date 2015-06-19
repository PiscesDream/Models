import matplotlib.pyplot as plt
from time import sleep
from numpy import *
import ga

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


#ref = http://blog.csdn.net/xuyuanfan/article/details/6726477
def tsp_crossover_f1(father, mother):
    length = len(father)
    cp = random.randint(0,length)
    child = []
    for ele in father[:cp+1]:
        child.append(ele)
    fa_genes = set(child)
    for ele in mother:
        if not (ele in fa_genes):
            child.append(ele)
#    print father,'+',mother,'->',child
    return child

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
    if random.rand()>0.4:
        return tsp_crossover_HGA(father, mather)
    else:
        return tsp_crossover_f1(father, mather)


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
        t = copy(base[posx])
        base[posx] = copy(base[posy])
        base[posy] = copy(t)
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

if __name__ == '__main__':
    global DataSize, data
    data = []
    for line in open('berlin52.tsp'):
        a = map(lambda x: float(x), line.strip().split(' '))
        data.append(array([a[1],a[2]]))
    data = array(data)

    DataSize = data.shape[0]
    createFactorialTable(DataSize)

    
#    def __init__(self, fitness_f, terminator, generation_size, genertaion_init, 
#        crossover_rate, crossover_f, mutation_rate, mutation_f, plot = False, plot_f = None):

#    fig = plt.figure()
    plt.ion()
    plt.show()
    
    MyGA = ga.GA(tsp_fitness_f, {'fitness_thresold':0}, 500, tsp_gen_init,
        0.9, tsp_crossover_HGA, 0.2, tsp_mutation_f, True, tsp_plot_f, 100, lambda a,b: a<b)
    MyGA.fit()

