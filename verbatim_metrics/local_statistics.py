#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

from verbatim_metrics.data import df, get_simulation_maps
from verbatim_metrics.plot import plot_dendrogram


def verbatim_intensity(index_map,
                       l=200,
                       b=10, # Window size around every pixel
                       fun=None,
                       post_fun=None,
                       threshold=0 # All pixels below this are set to zero
                       ):
    # Only works with square images for now
    l = index_map.shape[0]
    # Can possibly move this elsewhere for performance reasons
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
                fun = lambda a_index, b_original: (
                              np.sum(a_index == b_original) - 1) / ((b * 2 + 1) ** 2)

            verbatim[i, j] = fun(index_map[i - b:i + b + 1, j - b:j + b + 1],
                                 orig_map[r - b:r + b + 1, c - b:c + b + 1])
            if post_fun:
                verbatim[i, j] = post_fun(verbatim[i, j])
            if threshold > 0:
                verbatim[verbatim < threshold] = 0
            # Suprise verbatim[i, j] =  np.log(1/verbatim[i, j])
    verbatim = verbatim[b:-b, b:-b]
    return verbatim


# noinspection PyUnreachableCode
if False:
        #%% Testing
        training_image, sim_image, index_map = get_simulation_maps('stone',
                                                                   df.sample().iloc[0].at['simulation_parameters'])

        #%% Test the max verbatim possible in a random map
        scores = []
        for _ in range(100):
            random_map = np.random.randint(0,200**2,200**2).reshape((200,200))
            verbatim_map = verbatim_intensity(random_map)
            score = np.max(verbatim_map)
            scores.append(score)
        np.max(scores) # Result: max : 0.0065351, min: 0.00226
        #%% Test if this changes for different kernel sizes
        scores = []
        for kernel_size in np.arange(1,50):
            print(kernel_size)
            for _ in range(1):
                random_map = np.random.randint(0,200**2,200**2).reshape((200,200))
                verbatim_map = verbatim_intensity(random_map, b=kernel_size)
                score = np.max(verbatim_map)
                scores.append(score)
        np.max(scores),np.min(scores) # Result: max : 0.112, min: 0.00226
        #%% 50 50 image, see if we get 50% verbatim on average
        perfect_map = np.arange(200 ** 2).reshape((200,200))
        random_map = np.random.randint(0, 200 ** 2, 200 ** 2).reshape((200, 200))
        perfect_map[100:] = random_map[100:]
        scores = []
        for kernel_size in np.arange(1,10):
            print(kernel_size)
            verbatim_map = verbatim_intensity(perfect_map, b=kernel_size)
            score = np.mean(verbatim_map)
            scores.append(score)
        np.max(scores),np.min(scores) # Result: max : 0.112, min: 0.00226

        #%% 50% of the image is now random but with multiple edges
        perfect_map = np.arange(200 ** 2).reshape((200,200))
        random_map = np.random.randint(0, 200 ** 2, 200 ** 2).reshape((200, 200))
        b = 15
        scores = []
        bs = []
        expected = []
        for b in range(5,50,5):
            b=40
            for r in range(0, int(200/b), 2):
                for i in range(0,int(200/b), 2):
                    random_map[b*r:b*(r+1), b*i:b*(i+1)] = (
                        perfect_map[b*r:b*(r+1), b*i:b*(i+1)])
    #            perfect_map[:100, :100] = random_map[:100, :10*i]
            perfect_map = random_map
            plt.imshow(perfect_map)
            plt.show()
            # We expect to have 7*7*15*15 verbatim in a 200*200 map
            #
            # (7*7*15*15)/(200*200) = 0.2756
            expected.append()
            print(kernel_size)
            verbatim_map = verbatim_intensity(perfect_map, b=2)
            score = np.mean(verbatim_map)
            scores.append(score)
            bs.append(b)
            break
        np.max(scores),np.min(scores) # Result: max : 0.112, min: 0.00226
        np.argmax(scores) #-> 2
        #%% We can try the same with the circle maps
        maps = generate_synthetic_data()
        #%%
        scores = []
        for _, row in maps.iterrows():
            verbatim_map = verbatim_intensity(row.index_map, b=2)
            score = np.mean(verbatim_map)
            scores.append(score)
        np.max(scores), np.min(scores)  # Result: max : 0.112, min: 0.00226
        np.argmax(scores)  # -> 2
        # Average error -0.04
        np.mean(scores - maps.verbatim_percentage)
        # max error -> -0.1060
        np.min(scores - maps.verbatim_percentage)
        plt.plot(np.sort(scores - maps.verbatim_percentage))
        plt.show()


#%%
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
    training_image, sim_image, index_map = get_simulation_maps('strebelle',
                                                               df.sample().iloc[0].at['simulation_parameters'])
    #%%
    metrics = cluster_metric(index_map)
    metrics
    #%%
    plt.imshow(index_map)
    plt.show()
    #%%
    dicts_metrics = np.reshape(df['index_map'], -1).apply(cluster_metric)

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
