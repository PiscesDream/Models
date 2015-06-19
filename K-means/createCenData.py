from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as Ax

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

data = []
DATASIZE = 600
CEN = 6
SIGMA = 300
for i in range(CEN):
    x = random.randint(1, 3000)
    y = random.randint(1, 3000)
    z = random.randint(1, 3000)

    for t in xrange(DATASIZE/CEN):
        data.append([random.normal(x, SIGMA),
                     random.normal(y, SIGMA),
                     random.normal(z, SIGMA)])

save('data', data)

#print data
data = array(data)
ax.plot(data[:,0], data[:,1], data[:,2], color='orange', marker='o' ,ms=5, linestyle='None')
plt.show()
