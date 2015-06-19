from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as Ax

DATASIZE = 300;

def dec2hex(x):
    if (x==0):
        return '#000000'
    
    hexlist = '0123456789ABCDEF'
    s = ''
    ind = 6
    while (ind > 0):
        s += hexlist[x % 16]
        x /= 16
        ind -= 1
#    print '#'+s
    return '#'+s        

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

a = random.randint(1,1000,size=(DATASIZE, 3));
#print a
#print map(lambda x: dec2hex(x), a[:,1])

for (x, y, z) in zip(a[:,0], a[:,1], a[:,2]):
#    print line
    ax.plot([x], [y], [z], color='black', #dec2hex(int(log(x)+log(y)+log(z))), #dec2hex(random.randint(1,256*256*256)) 
        marker='o', linestyle='None')
plt.show()

save('Data', a)
