import aco
import plot
from numpy import *

if __name__ == '__main__':

    plot.plt_ion()
    plot.plt_show()
 
#    def __init__(self, nests, foods, obstacles,
#        pheromone_init==1, fade_rate, 
#        gen_speed, move_speed, sense_radius, life_long):
    
    myants = aco.ACO_module([[0,0],[40,40]], [[50, 50]], [[3,2]], 
        0.2,
        [0.3,0.1],  [1, 10],  [3 3],  [20, 3])
    myants.find_the_food()
