import matplotlib.pyplot as plt
from cPickle import load
from matplotlib.colors import ListedColormap as lcm

w, b = load(open('mnist_weight.dat','rb'))

w = w - w.min()
w = w * 16

print w.min()
print w.max()

fig, axs = plt.subplots(2, 5, sharex = True, sharey = True)
for i in xrange(2):
    for j in xrange(5):     
        axs[i,j].imshow(w[:, i*5+j].reshape(28, 28), cmap=lcm(['black', 'white']))
plt.show()

