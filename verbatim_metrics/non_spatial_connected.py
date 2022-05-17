import numpy as np
import matplotlib.pyplot as plt
from verbatim_metrics.data import df, get_simulation_maps


def verbatim_intensity(index_map, l=200, b=10, fun=None, post_fun=None):
    b = 10  # window size is b * 2 + 1
    l = 200
    orig_map = np.arange(l**2).reshape((l, l))
    orig_map = np.pad(orig_map, (b, b), mode='constant', constant_values=-1) #-1
    index_map = np.pad(index_map, (b, b), mode='constant', constant_values=-2) #-2
    # Single point
    verbatim = np.zeros_like(orig_map, dtype=np.double)
    for i in range(b, l + b):
        for j in range(b, l + b):
            (r, c) = np.divmod(index_map[i, j], l)
            r = r + b
            c = c + b
            # print(i, j)
            # print(r,c)
            # print(rand_map[i - b:i + b + 1, j - b:j + b + 1])
            # print(orig_map[r - b:r + b + 1, c - b:c + b + 1])
            if not fun:
                verbatim[i, j] = (np.sum(index_map[i - b:i + b + 1, j - b:j + b + 1]
                                         == orig_map[r - b:r + b + 1, c - b:c + b + 1]) - 1) / ((b*2+1)**2)
            else:
                verbatim[i, j] = fun(index_map[i - b:i + b + 1, j - b:j + b + 1],
                                    orig_map[r - b:r + b + 1, c - b:c + b + 1])
            if post_fun:
                verbatim[i, j] = post_fun(verbatim[i, j])
            # Suprise verbatim[i, j] =  np.log(1/verbatim[i, j])
    verbatim = verbatim[b:-b, b:-b]
    return verbatim


# %%
if __name__=='__main__':
    training_image, sim_image, index_map = get_simulation_maps('stone',
                                                               df.sample().iloc[0].at['simulation_parameters'])
    fig, axs = plt.subplots(2, 2)
    ax1 = plt.subplot(221)
    ax1.imshow(training_image)
    ax1.set_title('Training image')
    ax1.axis('off')
    ax2 = plt.subplot(222)
    ax2.imshow(sim_image)
    ax2.set_title('Simulation')
    ax2.axis('off')

    ax3 = plt.subplot(223)
    sourceIndex = np.stack(
        np.meshgrid(np.arange(training_image.shape[0]) / training_image.shape[0],
                    np.arange(training_image.shape[1]) / training_image.shape[1]) +
        [np.ones_like(training_image)],
        axis=-1);
    ax3.imshow(sourceIndex)
    ax3.imshow(sourceIndex.reshape(-1, 3)[index_map])
    ax3.set_title('Index map')
    ax3.axis('off')
    ax3 = plt.subplot(224)
    ax3.imshow(index_map)
    ax3.imshow(sourceIndex.reshape(-1, 3)[index_map])
    #ax3.imshow(label)
    plt.imshow(verbatim_intensity(index_map))
    ax3.set_title('Verbatim')
    ax3.axis('off')
    plt.show()
