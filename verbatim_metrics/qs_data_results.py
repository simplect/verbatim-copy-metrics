from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from datetime import datetime
import seaborn as sns

from timeit import default_timer as timer

from sklearn.linear_model import LinearRegression
from verbatim_metrics.global_statistics import global_metric
from verbatim_metrics.data import df_full, get_simulation_maps_n
from verbatim_metrics.local_statistics import cluster_metric, verbatim_metric, verbatim_intensity

def apply_metric(param_k, metric, samples=10, interval=1):
    print(f'starting thread {param_k} at {datetime.now().strftime("%H:%M")}')
    d = []
    results = []

    if samples is None:
        n_range = range(1,200,interval)
    else:
        n_range = range(samples)
    param_n = None
    for i in n_range:
        if samples is None:
            param_n = i
        maps_dict = get_simulation_maps_n('stone', param_k, param_n)

        d.append({'param': maps_dict['k'],
                  'param_n': maps_dict['n']})
        results.append(metric(maps_dict['icm']))
    df_params = pd.DataFrame(d)

    df_results = pd.DataFrame(results)
    return pd.concat([df_params, df_results], axis=1)


def applyParallel(params, func, metric, **kwargs):
    retLst = Parallel(n_jobs=4, prefer='threads')(delayed(func)(param, metric, **kwargs) for param in params)
    return pd.concat(retLst)

def applyParallelFull(params, func, metric, interval=1):
    retLst = Parallel(n_jobs=2, prefer='threads')\
                (delayed(func)(param, metric, None, interval) for param in params)
    return pd.concat(retLst)

#%% Results verbatim detection
print('Starting data collection')

params = [param for param in df_full['simulation_parameters'].unique()]
%time result = applyParallel(params, apply_metric, verbatim_metric, samples=50)
print('Done')
result['param'] = pd.to_numeric(result['param'])
result = result.sort_values('param')
#result.to_pickle('data/qs_verbatim_metric.pickle')

#result =  pd.read_pickle('data/qs_verbatim_metric.pickle')
#%% Figure 4c
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
fig, (ax1) = plt.subplots(1,1, figsize=(5,5))
p = sns.boxplot(data=result, x='param',
                    y='verbatim_estimated_percentage',color='seagreen' )

p.set(xlabel="QS parameter k",
      ylabel="Estimated percentage of pixels verbatim (V1)",
      ylim=(0,100),)
ax1.set_xticks(np.arange(0,15), ['1', 1.01, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5,
                                                 1.7, '2', 2.5, '3', '5', '10'])
plt.show()
#fig.savefig('img/qs_verbatim_metric.png', dpi=600, bbox_inches='tight')

#%% Figure 4a
#% Explore the data
with sns.axes_style('white'):
    #sim_image, realisation, index_map = get_simulation_maps('stone', '3.0')
    fig, ((ax0,ax1), (ax2, ax3)) = plt.subplots(2,2)
    p1 = ax0.imshow(sim_image, cmap='gray')
    ax0.axis('off')
    p1 = ax1.imshow(realisation, cmap='gray')
    ax1.axis('off')
    ax1.set_anchor('W')
    p1 = ax2.imshow(index_map, cmap='viridis')
    ax2.axis('off')
    sample_intensity = verbatim_intensity(index_map)
    p2 = ax3.imshow(sample_intensity)
    fig.colorbar(p2, ax=ax3, shrink=0.8)
    ax3.axis('off')
    ax3.set_anchor('W')
    fig.savefig('img/qs_input_intensity_output_low.png', dpi=600, bbox_inches='tight')
    plt.show()

#%% Cluster metric
print('Starting data collection')
params = [param for param in df_full['simulation_parameters'].unique()]
%time result = applyParallel(params, apply_metric, cluster_metric)
print('Done')
result['param'] = pd.to_numeric(result['param'])
result = result.sort_values('param')
#result.to_pickle('data/qs_cluster_metric.pickle')
result = pd.read_pickle('data/qs_cluster_metric.pickle')

#%% Figure 6
#result = pd.read_pickle('data/qs_cluster_metric.pickle')
result.mean_cluster_size = result.mean_cluster_size / result.num_clusters
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
fig, (ax1) = plt.subplots(1,1, figsize=(5,5))
p = sns.boxplot(data=result, x='param',
                    y='num_clusters', ax=ax1 )
p.set(xlabel="QS parameter k",
      ylabel="Found clusters (V2)")
ax1.set_xticks(np.arange(0,15), ['1', 1.01, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5,
                                                 1.7, '2', 2.5, '3', '5', '10'])
fig.savefig('img/qs_cluster_metric.png', dpi=600, bbox_inches='tight')
plt.show()
# Error of the found clusters
X = result.loc[:,'param'].to_numpy().reshape(-1,1)
y = result.loc[:,'num_clusters'].to_numpy().reshape(-1,1)
"{:.6f}".format(np.sum((X - y)**2) / X.shape[0])
reg = LinearRegression().fit(X,y)
reg.score(X,y)
#%% data for full collection
print('Starting data collection')

params = [param for param in df_full['simulation_parameters'].unique()]
%time result = applyParallelFull(params, apply_metric, verbatim_metric)
print('Done')

result['param'] = pd.to_numeric(result['param'])
result = result.sort_values('param')
#result.to_pickle('data/qs_verbatim_metric_full.pickle')
#%% Data for full collection clustering
print('Starting data collection')

params = [param for param in df_full['simulation_parameters'].unique()]
%time result = applyParallelFull(params, apply_metric, cluster_metric, interval=10)
print('Done')

result['param'] = pd.to_numeric(result['param'])
result = result.sort_values('param')
#result.to_pickle('data/qs_cluster_metric_full.pickle')
#%% Full analysis heatplot 6a
result = pd.read_pickle('data/qs_verbatim_metric_full.pickle')
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
data_pivot = (result
              .pivot('param_n', 'param', 'verbatim_estimated_percentage')
              .sort_index(axis=1, ascending=True)
              .sort_index(axis=0, ascending=False))
fig, (ax1) = plt.subplots(1,1, figsize=(9,6))
p = sns.heatmap(data=data_pivot, ax=ax1)
p.set(xlabel='QS parameter k', ylabel='QS parameter N')
#ax1.set_yticklabels(ax1.get_yticks(), rotation = 0)
#fig.savefig('img/qs_heatmap_k_n.png', dpi=600, bbox_inches='tight')
plt.show()
#%% Full analysis heatplot 6c
result = pd.read_pickle('data/qs_cluster_metric_full.pickle')
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
data_pivot = (result
              .pivot('param_n', 'param', 'num_clusters')
              .sort_index(axis=1, ascending=True)
              .sort_index(axis=0, ascending=False))
fig, (ax1) = plt.subplots(1,1, figsize=(9,6))
p = sns.heatmap(data=data_pivot, ax=ax1)
p.set(xlabel='QS parameter k', ylabel='QS parameter N')
#ax1.set_yticklabels(ax1.get_yticks(), rotation = 0)
fig.savefig('img/qs_heatmap_k_n_num_clusters.png', dpi=600, bbox_inches='tight')
plt.show()
#%% Full analysis heatplot 6b PCA
#%time result = collect_pca()
#np.save('data/mean_first_comp_qs.npy', result)
result = np.load('data/mean_first_comp_qs.npy')
params = [float(x) for x in df_full['simulation_parameters'].unique()]
result_df = pd.DataFrame.from_records(result.T, columns=params)
result_df = (result_df
             .sort_index(axis=1, ascending=True)
             .sort_index(axis=0, ascending=False))
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
#data_pivot = result.pivot('param', 'param_n', 'num_clusters')
fig, (ax1) = plt.subplots(1,1, figsize=(9,6))
p = sns.heatmap(data=result_df, ax=ax1)
p.set(xlabel='QS parameter k', ylabel='QS parameter N')
fig.savefig('img/qs_heatmap_k_n_PCA.png', dpi=600, bbox_inches='tight')
plt.show()

#%%
import seaborn as sns
sns.set_theme(style="ticks")
sns.pairplot(result, y_vars=[x for x in result.columns if x != 'u_patches' and x != 'index_map'], x_vars=['u_patches'])
plt.show()
