#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

from verbatim_metrics.data import df, get_simulation_maps
from verbatim_metrics.non_spatial_connected import verbatim_intensity
from verbatim_metrics.plot import plot_dendrogram


def verbatim_intensity(index_map, l=200, b=10, fun=None, post_fun=None):
    b = 10  # window size is b * 2 + 1
    l = 200
    orig_map = np.arange(l ** 2).reshape((l, l))
    orig_map = np.pad(orig_map, (b, b), mode='constant', constant_values=-1)  # -1
    index_map = np.pad(index_map, (b, b), mode='constant', constant_values=-2)  # -2
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
                                         == orig_map[r - b:r + b + 1, c - b:c + b + 1]) - 1) / ((b * 2 + 1) ** 2)
            else:
                verbatim[i, j] = fun(index_map[i - b:i + b + 1, j - b:j + b + 1],
                                     orig_map[r - b:r + b + 1, c - b:c + b + 1])
            if post_fun:
                verbatim[i, j] = post_fun(verbatim[i, j])
            # Suprise verbatim[i, j] =  np.log(1/verbatim[i, j])
    verbatim = verbatim[b:-b, b:-b]
    return verbatim


def cluster_metrics(input_map):
    verbatim = verbatim_intensity(input_map)
    X = np.reshape(verbatim, (-1, 1))
    X
    # Connectivity matrix based on the size of the input. So there is a connection for
    # every horizontal and vertical neighbor of a pixel.
    connectivity = grid_to_graph(input_map.shape[0], input_map.shape[1])

    threshold = 2.5

    model = AgglomerativeClustering(
        linkage="ward",
        connectivity=connectivity,
        compute_distances=True,
        compute_full_tree=True,
        distance_threshold=threshold,
        n_clusters=None

    )

    model.fit(X)
    label = np.reshape(model.labels_, input_map.shape[0:2])
    occurrence = np.unique(label, return_counts=True)
    # Remove the biggest cluster, that is the background
    if len(occurrence) < 2:
        raise Exception("No clusters, fix this")
    occ_index = occurrence[0].tolist()
    del occ_index[np.argmax(occurrence[1])]
    occ_count = occurrence[1].tolist()
    del occ_count[np.argmax(occurrence[1])]
    max_index = (input_map.shape[0] ** 2)
    metrics = {}
    metrics['mean_cluster_size'] = np.mean(occ_count) / max_index
    metrics['var_cluster_size'] = np.var(occ_count)
    metrics['num_clusters'] = len(occ_count)
    metrics['mean_cluster_size'] = metrics['mean_cluster_size'] * metrics['num_clusters']
    metrics['percent_verbatim'] = np.sum(verbatim) / max_index
    # Within cluster statistics
    flat_input = input_map.reshape(-1)
    flat_label = label.reshape(-1)

    metrics['within-cluster-range'] = (np.mean([np.max(flat_input[flat_label == i]) -
                                               np.min(flat_input[flat_label == i]) for i in occ_index])
                                      / max_index)
#    plot_dendrogram(model)
    return metrics


#%%
if __name__=='__main__':
    #%%
    training_image, sim_image, index_map = get_simulation_maps('stone',
                                                               df.sample().iloc[0].at['simulation_parameters'])
    #%%
    metrics = cluster_metrics(index_map)
    metrics
    #%%
    plt.imshow(index_map)
    plt.show()
    #%%
    dicts_metrics = np.reshape(df['index_map'], -1).apply(cluster_metrics)

    for x in dicts_metrics.items():
        for y in x[1].items():
            df.loc[x[0], y[0]] = y[1]

    # %%
    # Index(['training_image_type', 'simulation_parameters', 'index_map', 'mean',
    #        'variance', 'meanlog', 'variancelog', 'lincoef'],
    test_var = 'gradient'
    lowest = df.sort_values(test_var).iloc[1]
    highest = df.sort_values(test_var).iloc[-1]
    ax = plt.subplot(221)
    ax.imshow(lowest['index_map'])
    ax.axis('off')
    ax = plt.subplot(222)
    ax.imshow(highest['index_map'])
    ax.axis('off')

    ax = plt.subplot(223)
    # ax.imshow(np.diff(lowest['index_map'], axis=0))
    index_map = lowest['index_map']
    n = index_map.shape[0]
    lookup_y = np.arange(n ** 2).reshape((n, n)).T
    index_map_y = lookup_y[np.divmod(index_map, n)]
    gradient_l = (np.abs(((np.gradient(index_map)[1] + np.gradient(index_map_y)[0]) / 2)))
    gradient_l[gradient_l > 0] = 1 / gradient_l[gradient_l > 0]
    ax.imshow(gradient_l)
    ax.axis('off')

    ax = plt.subplot(224)
    # ax.imshow(np.diff(lowest['index_map'], axis=0))
    index_map = highest['index_map']
    n = index_map.shape[0]
    lookup_y = np.arange(n ** 2).reshape((n, n)).T
    index_map_y = lookup_y[np.divmod(index_map, n)]
    gradient = (np.abs(((np.gradient(index_map)[1] + np.gradient(index_map_y)[0]) / 2)))
    gradient[gradient > 0] = 1 / gradient[gradient > 0]
    ax.imshow(gradient)
    ax.axis('off')
    plt.show()

    # %%
    ax = plt.subplot(121)
    ax.imshow(index_map)
    ax.axis('off')
    ax = plt.subplot(122)
    filter_length = 4  # surrounding pixels to weight
    filter = [-1 / filter_length for i in range(int(filter_length / 2))] \
             + [1] + [-1 / filter_length for i in range(int(filter_length / 2))]
    corr_map = correlate(index_map, [[1 / 2, -1 / 2]], mode='constant', cval=0)
    corr_map[corr_map > 4] = 5
    # corr_map = corr_map / np.max(corr_map)
    plt.imshow(corr_map)
    plt.show()

    # %%
    # Reintrepret the index map to work with row direction.
    n = 200
    t = 4030
    lookup_y = np.arange(n ** 2).reshape((n, n)).T
    lookup_y = lookup_x.T
    np.divmod(index_map, n)
    index_map_y = lookup_y[np.divmod(index_map, n)]

    ax = plt.subplot(121)
    plt.imshow(index_map)
    ax = plt.subplot(122)
    plt.imshow(index_map_y)
    plt.show()
