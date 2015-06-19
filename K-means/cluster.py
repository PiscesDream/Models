from numpy import *
from scipy.cluster.vq import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ClustersNum = 6

def dec2hex(x):
    if (x==0):
        return '#000000'
    
    hexlist = '0123456789ABCDEF'
    s = ''
    for ind in xrange(6):
        s += hexlist[x % 16]
        x /= 16
#    print '#'+s
    return '#'+s        

datas = load('data.npy')
features = whiten(datas)

TRYTIMES = 10
CLUSTER = 6
test = False

if test:
    dif = [0]
    for i in xrange(TRYTIMES):
        dif.append(kmeans(features, i+1)[1])
        print i+1, '\t', dif[i+1]

    plt.plot(range(TRYTIMES+1),dif)
    plt.show()

clusters = map(lambda x: [], [[]]*CLUSTER)
centroids = kmeans(features, CLUSTER)[0]
print centroids
for indd, data in enumerate(features):
    dist = inf
    bel = -1
    for ind, val in enumerate(centroids):
#        print data,val
#        print data-val
#        print sum(data-val)
        if dist > sum((data - val)**2):
            dist = sum((data - val)**2)
            bel = ind
    clusters[bel].append(datas[indd])
#    print bel,

#print clusters[0]
save('Result', clusters)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

for son in clusters:
    sons = array(son)
    ax.plot(sons[:,0], sons[:,1], sons[:,2], color=dec2hex(random.randint(0, 256*256*256)), marker='o', ms=7, linestyle='-')
plt.show()
