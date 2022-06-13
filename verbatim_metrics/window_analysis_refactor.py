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
from verbatim_metrics.data import df_full, get_simulation_maps, get_simulation_maps_n, get_sim_handle
from verbatim_metrics.local_statistics import extract_verbatim_windows, verbatim_intensity
from verbatim_metrics.plot import plot_dendrogram
from verbatim_metrics.generate_data import generate_noisy_circles_map, generate_synthetic_data


#%% PCA Experiment RESULTS QS
# Sample verbatim windows then use PCA on it
def collect_pca():
    means_first_comp = np.zeros((15,200))
    num_samples = 200
    num_super_samples = 17
    b = 3
    bb = b * 2 + 1
    # Synthetic or qs
    d = []
    params = df_full['simulation_parameters'].unique()
    for current_super in range(num_super_samples):
        print(f'Super sample {current_super}')
        verbatim_windows = np.zeros((len(params), 200, num_samples, bb*bb))
        for i in range(len(params)):
            sim_handle = get_sim_handle('stone', params[i])
            for n in range(1, 200):
                realisation = get_simulation_maps_n('stone', params[i], n, sim_result=sim_handle)
                verbatim_windows[i,n,:,:] =\
                    extract_verbatim_windows(realisation['icm'],
                                             b=b,
                                             number_of_samples=num_samples).reshape(num_samples, -1)
        X = verbatim_windows.reshape(len(params)*200*num_samples,-1) > 0
        X[:, int((b * 2 + 1) ** 2 / 2)] = False  # Middle pixel zero
        pca = PCA(n_components=3)
        X_transformed = pca.fit_transform(X)
        sub_result = X_transformed.reshape(len(params), 200, num_samples, 3)
        sub_result_first_comp = sub_result[:,:,:,0]
        mean_first_comp = sub_result_first_comp.mean(axis=2)
        means_first_comp = means_first_comp + mean_first_comp
    means_first_comp = means_first_comp / num_super_samples
    return means_first_comp




#%%dd
for i in len(params):
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
    X_transformed = pca.fit_transform(X)
    np.mean(X_transformed[:, 0])
    ratios.append(pca.explained_variance_ratio_)
    first_transformed_comp.append(np.mean(np.dot(X, pca.components_[0,:])))
    params.append(param)
    #mean_intensity.append(np.mean(verbatim_intensities))
    d.append({'param': params,
              'ratios': ratios,
              'first_component_transformed': np.mean(np.dot(X, pca.components_[0,:])),
              'second_component_transformed': np.mean(np.dot(X, pca.components_[0,:])),
              'third_component_transformed': np.mean(np.dot(X, pca.components_[0,:])),
              'first_component': pca.components_[0, :].reshape(bb, bb),
              'second_component': pca.components_[1, :].reshape(bb, bb),
              'third_component': pca.components_[2, :].reshape(bb, bb)})
    break



results = pd.DataFrame()
#        results.to_pickle('data/windows.pickle')
results['param'] = pd.to_numeric(results['param'])
results = results.sort_values('param')
#    results.to_pickle('data/pca_qs_results.pickle')
#%% Figure 6a
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4))
p = sns.scatterplot(data=results, x='param',
                    y='first_component_transformed', ax=ax1)
p.set(xlabel="QS Parameter",
      ylabel="First PCA Component")
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
df = generate_synthetic_data(patches_range=100, samples_per_param=10, noise=0.40)

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
#    results.to_pickle('data/pca_qs_results.pickle')
#%% Figure 6c
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4))
p = sns.scatterplot(data=results, x='mean_intensity',
                    y='first_component_transformed', ax=ax1)
p.set(xlabel="Verbatim fraction",
      ylabel="First PCA component")
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
