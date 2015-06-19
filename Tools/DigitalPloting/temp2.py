import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as cm
import numpy as np

arr = np.random.randint(0,2,size=(10,20)).reshape(10,20)

#fig = plt.figure(figsize=(4, 4))
im = plt.imshow(arr, interpolation='none', vmin = 0, vmax = 1, cmap = cm(['white','black']))
plt.colorbar(im, use_gridspec=True)
plt.show()
