#%%
import time as time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.image import grid_to_graph
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from verbatim_metrics.data import df, get_simulation_maps
from verbatim_metrics.local_statistics import verbatim_intensity


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.show()

training_image, sim_image, index_map = get_simulation_maps('stone',
                                                           df.sample().iloc[0].at['simulation_parameters'])
plt.subplot(121)
plt.imshow(index_map)
plt.subplot(122)
plt.imshow(sim_image)
plt.show()
#%%
#np.save('data/training_low_verbatim',training_image)
#np.save('data/sim_low_verbatim',sim_image)
#np.save('data/index_low_verbatim', index_map)

#%% Dummy data
# perfect
index_map = np.arange(200*200).reshape((200,200))
# Fully random
index_map = np.random.randint(0,200*200,200*200).reshape((200,200))
index_map
#%% Preprocess image
def preprocess(training_image, index_map):
    # Translate indexes to 0-1 range
    rescale_index = np.stack(np.meshgrid(np.arange(training_image.shape[0]) / training_image.shape[0],
                                         np.arange(training_image.shape[1]) / training_image.shape[1]),
                             axis=-1)
    input_map = rescale_index.reshape(-1, 2)[index_map]
    input_map = gaussian_filter(input_map[:,:,0:2], sigma=0.7)
    return input_map
input_map = preprocess(training_image, index_map)
#%%
verbatim = verbatim_intensity(index_map)
X = np.reshape(verbatim, (-1, 1))
X
# Connectivity matrix based on the size of the input. So there is a connection for
# every horizontal and vertical neighbor of a pixel.
connectivity = grid_to_graph(input_map.shape[0], input_map.shape[1])

print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 30  # number of regions
ward = AgglomerativeClustering(
    n_clusters=n_clusters, linkage="ward", connectivity=connectivity, compute_distances=True, compute_full_tree=True
)
threshold = 3.5

ward = AgglomerativeClustering(
    linkage="ward",
    connectivity=connectivity,
    compute_distances=True,
    compute_full_tree=True,
    distance_threshold=threshold,
    n_clusters=None

)

ward.fit(X)
label = np.reshape(ward.labels_, input_map.shape[0:2])
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)
#%%
with sns.axes_style('white'):
    #sim_image, realisation, index_map = get_simulation_maps('stone', '3.0')

    fig, ((ax0,ax1,ax3)) = plt.subplots(1,3, figsize=(15,5))

    sourceIndex = np.stack(
        np.meshgrid(np.arange(training_image.shape[0]) / training_image.shape[0],
                    np.arange(training_image.shape[1]) / training_image.shape[1]) +
        [np.ones_like(training_image)],
        axis=-1);
    sim_image = sourceIndex.reshape(-1, 3)[index_map]
    p1 = ax0.imshow(sim_image, cmap='gray')
    ax0.axis('off')
    ax0.set_title('ICM')

    p1 = ax1.imshow(verbatim)
    ax1.axis('off')
    ax1.set_anchor('W')
    ax1.set_title('Verbatim density map')

    p2 = ax3.imshow(label, cmap='tab20')
    ax3.axis('off')
    ax3.set_anchor('W')
    ax3.set_title('Clusters')
    fig.savefig('img/qs_clusters.png', dpi=600, bbox_inches='tight')
    plt.show()
#%%

fig, axs = plt.subplots(2, 2)
ax2 = plt.subplot(221)
ax2.imshow(sim_image, cmap='gray')
ax2.set_title('Realisation')
ax2.axis('off')

ax3 = plt.subplot(222)

ax3.set_title('ICM')
ax3.axis('off')
ax3 = plt.subplot(223)
ax3.imshow(index_map)
ax3.imshow(sourceIndex.reshape(-1, 3)[index_map])
#ax3.imshow(label)
plt.imshow(label)
#%%

fig, ax0 = plt.subplots(1, 1, figsize=(15, 3))
model = ward
truncate_mode="level"
p=7
color_threshold=threshold
counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)
for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count

linkage_matrix = np.column_stack(
    [model.children_, model.distances_, counts]
).astype(float)

# Plot the corresponding dendrogram
dendrogram(linkage_matrix, p=p, color_threshold=color_threshold, truncate_mode=truncate_mode)
fig.savefig('img/qs_dendogram.png', dpi=600, bbox_inches='tight')
#%%
model = ward

counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)
# for every merge
for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples] # add the count of the lower cluster
    counts[i] = current_count

linkage_matrix = np.column_stack(
    [model.children_, model.distances_, counts]
).astype(float)
#[(i, merge) for i, merge in enumerate(model.children_)][-1]
# Plot the corresponding dendrogram

df = pd.DataFrame({'distances':model.distances_, 'count':counts})
sns.set_theme(style="darkgrid")
g = sns.relplot(x="distances", y="count", data=df)
g.set(xlim=(0,60))
plt.show()
#%%
metrics = {}
metrics['mean_cluster_distance'] = np.mean(model.distances_ * counts)
metrics['verbatim_merges'] = np.sum(model.distances_ >= 10)
metrics

#%%
for i in range(model.n_clusters_):
    masked = input_map[label == i,:]
    print(i)
masked
#%%
for i, merge in :
    if counts[i] > 20000:
        for l in range(ward.n_clusters_):
            plt.contour(
                label == l,
                colors=[
                    plt.cm.nipy_spectral(l / float(ward.n_clusters_)),
                ],
                linewidths=0.5
            )

