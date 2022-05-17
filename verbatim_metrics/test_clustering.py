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
from verbatim_metrics.non_spatial_connected import verbatim_intensity
from verbatim_metrics.plot import plot_dendrogram

training_image, sim_image, index_map = get_simulation_maps('stone',
                                                           df.sample().iloc[0].at['simulation_parameters'])
verbatim = verbatim_intensity(index_map)

plt.subplot(221)
plt.imshow(index_map)
plt.subplot(222)
plt.imshow(sim_image)
plt.subplot(223)
plt.imshow(verbatim)
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

#%%
input_map = verbatim
X = np.reshape(verbatim, (-1,1))
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
threshold = 2.5

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


fig, axs = plt.subplots(2, 2)
ax1 = plt.subplot(221)
ax1.imshow(training_image)
ax1.set_title('Training image')
ax1.axis('off')
ax2 = plt.subplot(222)
ax2.imshow(verbatim)
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
ax3.set_title('Index origin')
ax3.axis('off')
ax3 = plt.subplot(224)
ax3.imshow(index_map)
ax3.imshow(sourceIndex.reshape(-1, 3)[index_map])
#ax3.imshow(label)
plt.imshow(label)
def plot_cluster_contours(ward):
    for l in range(ward.n_clusters_):
        plt.contour(
            label == l,
            colors=[
                plt.cm.nipy_spectral(l / float(ward.n_clusters_)),
            ],
            linewidths=0.5
        )
ax3.set_title('Index map')
ax3.axis('off')
plt.show()

plot_dendrogram(ward, truncate_mode="level",  p=8, color_threshold=threshold)

#%%


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
#sns.set_theme(style="darkgrid")
g = sns.relplot(x="distances", y="count", data=df)
g.set(xlim=(0,60))
plt.show()
#%%
occurrence = np.unique(label, return_counts=True)
# Remove the biggest cluster, that is the background
occ_index = occurrence[0].tolist()
del occ_index[np.argmax(occurrence[1])]
occ_count = occurrence[1].tolist()
del occ_count[np.argmax(occurrence[1])]
metrics = {}
metrics['mean_cluster_distance'] = np.mean(model.distances_ * counts)
metrics['verbatim_merges'] = np.sum(model.distances_ >= 7)
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

