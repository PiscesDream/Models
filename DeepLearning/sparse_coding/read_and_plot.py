import scipy.io as sio
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_dig(data, ax, W = None, H = None):
    if W == None:
        W = int(len(data) ** .5)
        H = W
    ax.imshow(data.reshape(H,W), interpolation='none', cmap = cm.gray)       

def rand_plot(data, row = 5, col = 5):
    fig, axs = plt.subplots(row, col, sharex = True, sharey = True)
    for i in xrange(row):
        for j in xrange(col):     
            ind = random.randint(0, data.shape[0])
            plot_dig(data[ind, :], axs[i,j])
    plt.show()
    
def all_plot(data, row = None, col = None):
    l = int(len(data)**0.5)
    fig, axs = plt.subplots(l, l, sharex = True, sharey = True)
    for i in xrange(l):
        for j in xrange(l):
            ind = i*l + j
            plot_dig(data[ind], axs[i,j])
    plt.show()


def load_data(num, dim):
#===========================loading=========================

    data = sio.loadmat('IMAGES.mat')['IMAGES']


#==========================pic=============================

    patch_size = dim
    sample_size = num
    samples = zeros((sample_size, patch_size * patch_size))

    for i in xrange(sample_size):
        ind = random.randint(0, 10)

        x = random.randint(0, 512 - patch_size)
        y = random.randint(0, 512 - patch_size)

    samples[i, :] = data[x:x+patch_size, y:y+patch_size, ind].flatten()
#========================normalize=========================
#    % Squash data to [0.1, 0.9] since we use sigmoid as the activation
#    % function in the output layer
#
#    % Remove DC (mean of images). 
#    patches = bsxfun(@minus, patches, mean(patches));        
#
#    % Truncate to +/-3 standard deviations and scale to -1 to 1
#    pstd = 3 * std(patches(:));
#    patches = max(min(patches, pstd), -pstd) / pstd;
#
#    % Rescale from [-1,1] to [0.1,0.9]
#    patches = (patches + 1) * 0.4 + 0.1;
    samples = samples - samples.mean(0)
    pstd = 3 * samples.flatten().std(ddof = 1)
    for i in xrange(samples.shape[0]):
        for j in xrange(samples.shape[1]):
            samples[i, j] = max(min(samples[i, j], pstd), -pstd) / pstd;
    samples = (samples + 1) * 0.4 + 0.1;                                      
            
       
#===========================plot============================

    rand_plot(samples, 2, 2)        
#    all_plot(samples)

    return samples
