import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from verbatim_metrics.data import df, get_simulation_maps
from verbatim_metrics.plot import plot_dendrogram


def verbatim_intensity(index_map,
                       b=3,  # Window size around every pixel
                       fun=None,
                       threshold=0  # All pixels below this are set to zero
                       ):
    # Only works with square images for now
    l = index_map.shape[0]
    # Can possibly move this elsewhere for performance reasons
    orig_map = np.arange(l ** 2).reshape((l, l))
    verbatim = np.zeros(l ** 2)
    #    verbatim = np.zeros_like(orig_map, dtype=np.double)
    orig_map = np.pad(orig_map, (b, b), mode='constant', constant_values=-1)  # -1
    index_map = np.pad(index_map, (b, b), mode='constant', constant_values=-2)  # -2
    bw = b * 2 + 1
    index_sw = sliding_window_view(index_map, (bw, bw)).reshape(-1, bw, bw)
    orig_sw = sliding_window_view(orig_map, (bw, bw)).reshape(-1, bw, bw)
    # Single point
    for n in np.arange(l ** 2):
        if not fun:
            fun = lambda a_index, b_original: (np.sum(a_index == b_original) - 1) / (bw ** 2)
        verbatim[n] = fun(index_sw[n],
                          orig_sw[index_sw[n][b, b]])

        if threshold > 0:
            verbatim[verbatim < threshold] = 0
    return verbatim.reshape(l, l)


def extract_verbatim_windows(index_map,
                             b=3,  # Pixels around the pixel in horizontal and vertical direction
                             activation_function=None,  # Optional alternative function for calculating the activation
                             sampling_rate=1,  #Randomly sample windows
                             number_of_samples=None
                             ):
    # Only works with square images for now
    index_map_height = index_map.shape[0]
    number_of_pixels = index_map_height ** 2
    window_size = b * 2 + 1

    # Can possibly move this elsewhere for performance reasons
    perfect_map = np.arange(number_of_pixels).reshape((index_map_height, index_map_height))
    perfect_map_padded = np.pad(perfect_map, (b, b), mode='constant', constant_values=-1)  # -1
    index_map = np.pad(index_map, (b, b), mode='constant', constant_values=-2)  # -2
    # Single point

    index_sw = sliding_window_view(index_map,
                                   (window_size, window_size)).reshape(-1, window_size, window_size)
    
    orig_sw = sliding_window_view(perfect_map_padded,
                                  (window_size, window_size)).reshape(-1, window_size, window_size)
    if sampling_rate == 1:
        window_indexes = np.arange(number_of_pixels)
    else:
        window_indexes = np.random.randint(0, number_of_pixels, int(number_of_pixels * sampling_rate))

    if number_of_samples is not None:
        window_indexes = np.random.choice(window_indexes, number_of_samples)

    verbatim_windows = np.zeros((len(window_indexes), window_size, window_size))
    for window_index in range(len(window_indexes)):
        if not activation_function:
            activation_function = lambda a_index, b_original: a_index * (a_index == b_original)
        sliding_window_index = window_indexes[window_index]
        verbatim_windows[window_index, :, :] = activation_function(index_sw[sliding_window_index],
                                                           orig_sw[index_sw[sliding_window_index][b, b]])

    return verbatim_windows


def pca_metrics():
    test = extract_verbatim_windows(index_map, b=10)
    freq = np.sum(test > 0, axis=0)
    freq
    plt.imshow(freq)
    plt.show()
    # %%
    X = test.reshape(test.shape[0], -1) > 0
    #        X = test.reshape(test.shape[0],-1)
    X[:, 220] = False

    pca = PCA(n_components=10)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    plt.title(np.round(pca.explained_variance_ratio_[0], 5))
    plt.imshow(pca.components_[0].reshape(21, 21))


def verbatim_metric(input_map):
    verbatim = verbatim_intensity(input_map)
    max_index = (input_map.shape[0] ** 2)
    metrics = {}
    metrics['verbatim_estimated_percentage'] = (np.sum(verbatim > 0)) / max_index * 100
    return metrics


# %%
def cluster_metric(input_map):
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
    metrics['estimated_verbatim_percentage'] = np.sum(occ_count) / max_index
    metrics['var_cluster_size'] = np.var(occ_count)
    metrics['num_clusters'] = len(occ_count)
    metrics['mean_cluster_size'] = metrics['mean_cluster_size'] * metrics['num_clusters']
    metrics['percent_verbatim'] = (np.sum(verbatim > 0) / max_index)
    # Within cluster statistics
    flat_input = input_map.reshape(-1)
    flat_label = label.reshape(-1)

    metrics['within-cluster-range'] = (np.mean([np.max(flat_input[flat_label == i]) -
                                                np.min(flat_input[flat_label == i]) for i in occ_index])
                                       / max_index)
    #    plot_dendrogram(model)
    return metrics

