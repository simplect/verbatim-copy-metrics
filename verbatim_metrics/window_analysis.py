# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import seaborn as sns

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from verbatim_metrics.data import df_full,df, get_simulation_maps
from verbatim_metrics.local_statistics import extract_verbatim_windows, verbatim_intensity
from verbatim_metrics.plot import plot_dendrogram
from verbatim_metrics.generate_data import generate_noisy_circles_map, generate_synthetic_data


# %% PCA Experiment RESULTS QS
# Sample verbatim windows then use PCA on it
samples = 100
params = []
ratios = []
first_transformed_comp = []
second_transformed_comp = []
third_transformed_comp = []
first_component = []
second_component = []
third_component = []
mean_intensity = []
b = 3
bb = b * 2 + 1
# Synthetic or qs

for param in df_full['simulation_parameters'].unique():
    print(param)
    ratios_sample = []
    index_maps = [get_simulation_maps('stone', param)[2] for _ in range(samples)]
    #verbatim_intensities = [np.sum(verbatim_intensity(index_map) > 0)/(200**2)
    #                                   for index_map in index_maps]
    verbatim_windows = np.concatenate([extract_verbatim_windows(index_map,
                                                                b=b,
                                                                sampling_rate=0.15)
                                       for index_map in index_maps],
                                      axis=0)
    X = verbatim_windows.reshape(verbatim_windows.shape[0], -1) > 0
    X[:, int((b*2+1)**2/2)] = False  # Middle pixel zero
    pca = PCA(n_components=3)
    pca.fit(X)

    ratios.append(pca.explained_variance_ratio_)
    first_transformed_comp.append(np.mean(np.dot(X, pca.components_[0,:])))
    params.append(param)
    #mean_intensity.append(np.mean(verbatim_intensities))
    first_component.append(pca.components_[0, :].reshape(bb, bb))
    second_component.append(pca.components_[1, :].reshape(bb, bb))
    third_component.append(pca.components_[2, :].reshape(bb, bb))

results = pd.DataFrame({'param': params,
                        'ratios': ratios,
                        'first_component_transformed': first_transformed_comp,
                        'first_component': first_component,
                        'second_component': second_component,
                        'third_component': third_component,
    #                    'mean_intensity': mean_intensity
                                            })
#        results.to_pickle('data/windows.pickle')
results['param'] = pd.to_numeric(results['param'])
results = results.sort_values('param')
#    results.to_pickle('data/pca_qs_results.pickle')
#%% Figure 6a
results = pd.read_pickle('data/pca_qs_results.pickle')
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
p = sns.scatterplot(data=results, x='param',
                    y='first_component_transformed', ax=ax1)
p.set(xlabel="QS parameter k",
      ylabel="PCA metric (V4)")
ax1.set_yscale('log', base=2)
#ax1.set_xticks(np.arange(0,15), ['1', 1.01, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5,
#                                                 1.7, '2', 2.5, '3', '5', '10'])
#p = sns.scatterplot(data=results, x='param',
#                    y='mean_intensity', ax=ax2)
#p.set(xlabel="QS Parameter",
#      ylabel="Fraction of verbatim pixels")
#plt.show()

plt.savefig('img/pca_transformed_qs.png', dpi=600, bbox_inches='tight')

# %% PCA Experiment RESULTS synthetic
# Sample verbatim windows then use PCA on it
samples = 100
params = []
ratios = []
first_transformed_comp = []
second_transformed_comp = []
third_transformed_comp = []
first_component = []
second_component = []
third_component = []
mean_intensity = []
b = 3
bb = b * 2 + 1
# Synthetic or qs
df = generate_synthetic_data(patches_range=600, samples_per_param=10, noise=0.40)

for param, group in df.groupby(df.u_patches):
    ratios_sample = []
    verbatim_windows = np.concatenate([extract_verbatim_windows(index_map,
                                                                b=b,
                                                                sampling_rate=0.25)
                                       for index_map in group.index_map],
                                      axis=0)
    verbatim_percentage = group.verbatim_percentage.mean()
    X = verbatim_windows.reshape(verbatim_windows.shape[0], -1) > 0
    X[:, int((b*2+1)**2/2)] = False  # Middle pixel zero
    pca = PCA(n_components=3)
    pca.fit(X)

    ratios.append(pca.explained_variance_ratio_)
    first_transformed_comp.append(np.mean(np.dot(X, pca.components_[0,:])))
    print(np.mean(np.dot(X, pca.components_[0,:])))
    params.append(param)
    first_component.append(pca.components_[0, :].reshape(bb, bb))
    mean_intensity.append(verbatim_percentage)

results = pd.DataFrame({'param': params,
                        'ratios': ratios,
                        'first_component_transformed': first_transformed_comp,
                        'first_component': first_component,
                        'mean_intensity': mean_intensity})
#        results.to_pickle('data/windows.pickle')
results['param'] = pd.to_numeric(results['param'])
results = results.sort_values('param')
results.to_pickle('data/synth_qs_results.pickle')
#%% Figure 6c
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
p = sns.scatterplot(data=results, x='mean_intensity',
                    y='first_component_transformed', ax=ax1)
p.set(xlabel="Percentage of pixels verbatim (V1)",
      ylabel="PCA Metric (V4)")
ax1.yaxis.set_ticks(np.arange(0, 2.1, 0.25))
plt.plot([0, 80], [0,2], linewidth=1, color='green')
#plt.show()

plt.savefig('img/pca_transformed_synth.png', dpi=600, bbox_inches='tight')


#%% Plot for results Figure 6b
results = results.sort_values('param')
with sns.axes_style('white'):
    fig, (axis, axis2, axis3) = plt.subplots(3, 15)
    for i in range(15):
        p1 = axis[i].imshow(results.iat[i, 3], cmap='viridis')
        axis[i].set_title(results.iat[i, 0])
        axis[i].axis('off')
    for i in range(15):
        p1 = axis2[i].imshow(results.iat[i, 4], cmap='viridis')
        axis2[i].set_title(results.iat[i, 0])
        axis2[i].axis('off')
    for i in range(15):
        p1 = axis3[i].imshow(results.iat[i, 5], cmap='viridis')
        axis3[i].set_title(results.iat[i, 0])
        axis3[i].axis('off')
    #fig.savefig('img/first_components_synth.png', dpi=600, bbox_inches='tight')
    plt.show()

# %%
df['simulation_parameters'].unique()

# %% Testing frequency windows
index_map_n, verbatim = generate_noisy_circles_map(100)
index_map, verbatim = generate_noisy_circles_map(10, perfect_circles=True, noise=0.6)
#    training_image, sim_image, index_map = get_simulation_maps('stone',
#                                                               df.sample().iloc[0].at['simulation_parameters'])
# %%
test = pattern_freq(index_map_n, b=10)
X_n = test.reshape(test.shape[0], -1) > 0
X_n[:, 220] = False
test = pattern_freq(index_map, b=10)
# X = test.reshape(test.shape[0],-1) > 0
# X[:,220] = False
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)
# print(pca.singular_values_[0])
plt.subplot(232)
plt.imshow(index_map)
plt.title('index origin')
plt.axis('off')
plt.subplot(234)
plt.title(np.round(pca.explained_variance_ratio_[0], 5))
plt.imshow(pca.components_[0].reshape(21, 21))
plt.axis('off')
plt.subplot(235)
plt.title(np.round(pca.explained_variance_ratio_[1], 5))
plt.imshow(pca.components_[1].reshape(21, 21))
plt.axis('off')
plt.subplot(236)
plt.title(np.round(pca.explained_variance_ratio_[2], 5))
plt.imshow(pca.components_[2].reshape(21, 21))
plt.axis('off')
plt.suptitle(f'verbatim: {verbatim}')
plt.show()

fc = pca.transform(X)[:, 0]
plt.imshow(fc.reshape(200, 200))
plt.show()
#%% test

#training_image, sim_image, index_map = get_simulation_maps('stone', df.sample().iloc[0].at['simulation_parameters'])
plt.subplot(131)
plt.imshow(training_image)
plt.axis('off')
plt.subplot(132)
plt.imshow(sim_image)
plt.axis('off')
plt.subplot(133)
plt.imshow(verbatim_intensity(index_map))
plt.axis('off')
#plt.show()
plt.savefig('img/verbatim_density.png',dpi=600, bbox_inches='tight')
