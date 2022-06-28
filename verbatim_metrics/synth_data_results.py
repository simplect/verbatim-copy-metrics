from joblib import Parallel, delayed
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression

from verbatim_metrics.generate_data import generate_synthetic_data
from verbatim_metrics.global_statistics import global_metric
from verbatim_metrics.local_statistics import cluster_metric, verbatim_metric, verbatim_intensity


def apply_metric(df, metric):
    print(f'starting thread at {datetime.now().strftime("%H:%M")}')
    df_metrics = pd.DataFrame(list(df['index_map'].apply(metric)), index=df.index)
    df = df.drop(columns=['index_map'])
    return pd.concat([df, df_metrics], axis=1)

def apply_metric_old(df, metric):
    print(f'starting thread at {datetime.now().strftime("%H:%M")}')
    dicts_metrics = df['index_map'].apply(metric)

    for x in dicts_metrics.items():
        for y in x[1].items():
            df.loc[x[0], y[0]] = y[1]
    return df

def applyParallel(dfGrouped, func, metric):
    retLst = Parallel(n_jobs=4, prefer='threads')(delayed(func)(group, metric) for name, group in dfGrouped)
    return pd.concat(retLst)


# %% Methods 3.1 Test on full noise maps.
scores = []
for _ in range(100):
    random_map = np.random.randint(0, 200 ** 2, 200 ** 2).reshape((200, 200))
    verbatim_map = verbatim_intensity(random_map)
    score = np.sum(verbatim_map > 0) / (200**2)
    scores.append(score)
np.min(scores), np.max(scores), np.mean(scores)
# %% Test if this changes for different kernel sizes
scores = []
times = []
sizes = []
random_map = np.random.randint(0, 200 ** 2, 200 ** 2).reshape((200, 200))
for kernel_size in np.arange(1, 50):
    print(kernel_size)
    sizes.append(kernel_size)
    start = timer()
    verbatim_map = verbatim_intensity(random_map, b=kernel_size)
    end = timer()
    times.append(end-start)
    print(end - start)  # Time in seconds, e.g. 5.38091952400282
    score = np.sum(verbatim_map > 0) / (200 ** 2)
    scores.append(score)
np.min(scores), np.max(scores), np.mean(scores)

scores_ksize = scores

#%% Calibrate window size on perfect squares
perfect_map = np.arange(200 ** 2).reshape((200, 200))
random_map = np.random.randint(0, 200 ** 2, 200 ** 2).reshape((200, 200))
zeros_map = np.zeros_like(random_map)
b = 15
scores = []
bs = []
b = 40
for r in range(0, int(200 / b), 2):
    for i in range(0, int(200 / b), 2):
        random_map[b * r:b * (r + 1), b * i:b * (i + 1)] = (
            perfect_map[b * r:b * (r + 1), b * i:b * (i + 1)])
        zeros_map[b * r:b * (r + 1), b * i:b * (i + 1)] = (
            perfect_map[b * r:b * (r + 1), b * i:b * (i + 1)])
#            perfect_map[:100, :100] = random_map[:100, :10*i]
perfect_map = random_map
plt.imshow(perfect_map)
plt.show()
np.sum(zeros_map > 0)/(200**2) # Thruth is 0.359975
for b in range(1, 50, 1):
    # We expect to have 7*7*15*15 verbatim in a 200*200 map
    #
    # (7*7*15*15)/(200*200) = 0.2756
    print(b)
    verbatim_map = verbatim_intensity(perfect_map, b=b)
    score = np.sum(verbatim_map > 0) / (200 ** 2)
    scores.append(score)
    bs.append(b)
np.min(scores), np.max(scores), np.mean(scores)
scores_perfect_squares = scores - (np.sum(zeros_map > 0)/(200**2))

#%% Plot previous three results
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
x = [i*2+1 for i in range(1,50)]
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,4))
p = sns.scatterplot(x=x,
                    y=times,
                    ax=ax1)
p.set(xlabel="Window size",
      ylabel="Runtime in seconds",
      xlim=(1, 41),
      ylim=(0,1),
      xticks=x[0:20])
p = sns.scatterplot(x=x,
                    y=scores_ksize,
                    ax=ax2)
p.set(xlabel="Window size",
      ylabel="Verbatim fraction detected in noise",
      xlim=(1, 41),
      ylim=(0, 0.05),
      xticks=x[0:20])
p = sns.scatterplot(x=x,
                    y=scores_perfect_squares,
                    ax=ax3)
p.set(xlabel="Window size",
      ylabel="Error predicting verbatim in squares",
      xlim=(1,41),
      ylim=(0, 0.025),
      xticks=x[0:20])
ax1.set_xticklabels(ax1.get_xticks(), rotation = 90)
ax2.set_xticklabels(ax2.get_xticks(), rotation = 90)
ax3.set_xticklabels(ax3.get_xticks(), rotation = 90)
plt.show()
#fig.savefig('img/window_size_tuning.png', dpi=600, bbox_inches='tight')

#%%

#%% Results 1
# Either this or read the pickle below
df = generate_synthetic_data(patches_range=600, samples_per_param=10, noise=0.40)
print('Starting data collection')
%time result = applyParallel(df.groupby(df.u_patches), apply_metric, verbatim_metric)
print('Done')

#result.to_pickle('data/synth_verbatim_metric.pickle')
#result = pd.read_pickle('data/synth_verbatim_metric.pickle')
#%% Figure 3
def plot_sim_index(index_map):
    sourceIndex = np.stack(
        np.meshgrid(np.arange(index_map.shape[0]) / index_map.shape[0],
                    np.arange(index_map.shape[1]) / index_map.shape[1]) +
        [np.ones_like(index_map)],
        axis=-1)
    return sourceIndex.reshape(-1, 3)[index_map]
with sns.axes_style('white'):
    fig, (ax0,ax1, ax2) = plt.subplots(1,3)
    i = 00
    index_map = df.iloc[i,1]
    verbatim_percentage = np.round(df.iloc[i,2],2)
    p1 = ax0.imshow(plot_sim_index(index_map), cmap='viridis')
    ax0.set_title(f'{verbatim_percentage}')
    ax0.axis('off')
    i = 800
    index_map = df.iloc[i, 1]
    verbatim_percentage = np.round(df.iloc[i,2],2)
    ax1.set_title(f'{verbatim_percentage}')
    p1 = ax1.imshow(plot_sim_index(index_map), cmap='viridis')
    ax1.axis('off')
    i = 5000
    index_map = df.iloc[i, 1]
    verbatim_percentage = np.round(df.iloc[i,2],2)
    ax2.set_title(f'{verbatim_percentage}')
    p1 = ax2.imshow(plot_sim_index(index_map), cmap='viridis')
    ax2.axis('off')
    fig.savefig('img/noisy_circles_crop.png', dpi=326, bbox_inches='tight')
    plt.show()
#%% Figure 4b
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
fig, (ax1) = plt.subplots(1,1, figsize=(6,6))
p = sns.scatterplot(data=result, x='verbatim_percentage',
                    y='verbatim_estimated_percentage' )
p.set(xlabel="Percentage of pixels verbatim (V1)",
      ylabel="Estimated percentage of pixels verbatim (V1)",
      xlim=(0,100),
      ylim=(0,100))
plt.show()
#fig.savefig('img/synth_verbatim_metric.png', dpi=600, bbox_inches='tight')
#%% Linear regression and MSE result
X = result.loc[:,'verbatim_percentage'].to_numpy().reshape(-1,1)
y = result.loc[:,'verbatim_estimated_percentage'].to_numpy().reshape(-1,1)
reg = LinearRegression().fit(X,y)
reg.score(X,y)
np.sum((X - y)**2) / X.shape[0]
"{:.6f}".format(np.sum((X - y)**2) / X.shape[0])
#%% Figure 4a
#% Explore the data
sample = result.sample()
with sns.axes_style('white'):
    fig, (ax1, ax2) = plt.subplots(1,2)
    p1 = ax1.imshow(sample.iloc[0].at['icm'], cmap='viridis')
    ax1.axis('off')
    fig.colorbar(p1, ax=ax1, shrink=0.45)
    sample_intensity = verbatim_intensity(sample.iloc[0].at['icm'])
    p2 = ax2.imshow(sample_intensity)
    fig.colorbar(p2, ax=ax2, shrink=0.45)
    ax2.axis('off')
    #fig.savefig('img/noise_input_intensity_output.png', dpi=600, bbox_inches='tight')
    plt.show()

#%% Cluster metrics
def apply_cluster_metric(df, metric=cluster_metric):
    print(f'Num patches {df.iloc[0,0]}')
    return apply_metric(df, metric=metric)
df = generate_synthetic_data(patches_range=100, samples_per_param=5, noise=0.40)
print('Starting data collection')
%time result = applyParallel(df.groupby(df.u_patches), apply_cluster_metric)
print('Done')

#result.to_pickle('data/synth_cluster_metric.pickle')
#%%
result = pd.read_pickle('data/synth_cluster_metric.pickle')
result.mean_cluster_size = result.mean_cluster_size / result.num_clusters
result.verbatim_percentage = result.verbatim_percentage * 100
result.mean_cluster_size = result.mean_cluster_size *100
#%% Figure 6
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(17,5))
p = sns.scatterplot(data=result, x='n_patches',
                    y='num_clusters', ax=ax1 )
p.set(xlabel="Number of verbatim patches",
      ylabel="Number of verbatim clusters (V2)",
      ylim=(-1.5,75),
      xlim=(-1.5,75))
ax1.axline([0,0],[80,80], color='green')
p = sns.scatterplot(data=result, x='verbatim_percentage',
                    y='num_clusters', ax=ax2 )
p.set(xlabel="Percentage of pixels verbatim (truth)",
      ylabel="Number of verbatim clusters (V2)")
ax2.axline([0,0],[50,50], color='green')
p = sns.scatterplot(data=result, x='verbatim_percentage',
                    y='mean_cluster_size', ax=ax3)
p.set(xlabel="Percentage of pixels verbatim (truth)",
      ylabel="Average verbatim cluster size % (V3)")
fig.savefig('img/synth_cluster_metric.png', dpi=600, bbox_inches='tight')
#plt.show()
# Error of the found clusters
X = result.loc[:,'n_patches'].to_numpy().reshape(-1,1)
y = result.loc[:,'num_clusters'].to_numpy().reshape(-1,1)
"{:.6f}".format(np.sum((X - y)**2) / X.shape[0])
reg = LinearRegression().fit(X,y)
reg.score(X,y)


#%%
# Index(['training_image_type', 'simulation_parameters', 'index_map', 'mean',
#        'variance', 'meanlog', 'variancelog', 'lincoef'],
test_var = 'verbatim_percentage'
lowest = df.sort_values(test_var).iloc[0]
highest  = df.sort_values(test_var).iloc[-1]
ax = plt.subplot(221)
ax.imshow(lowest['index_map'])
ax.axis('off')
ax = plt.subplot(222)
ax.imshow(highest['index_map'])
ax.axis('off')

ax = plt.subplot(223)
index_map_1d = lowest['index_map'].reshape(-1)
X = (np.sort(np.histogram(index_map_1d, bins=len(index_map_1d))[0]))

ax.set_ylim(0,20)
ax.plot(X)


ax = plt.subplot(224)
index_map_1d = highest['index_map'].reshape(-1)
X = (np.sort(np.histogram(index_map_1d, bins=len(index_map_1d))[0]))

ax.set_ylim(0,20)
ax.plot(X)
plt.show()

#%%
import seaborn as sns
sns.set_theme(style="ticks")
sns.pairplot(result, y_vars=[x for x in result.columns if x != 'u_patches' and x != 'index_map'], x_vars=['u_patches'])
plt.show()


#%%
# noinspection PyUnreachableCode
# %% Testing
#training_image, sim_image, index_map = get_simulation_maps('stone',
 #                                                          df.sample().iloc[0].at['simulation_parameters'])


# %%
if __name__ == '__main__':
# %%
training_image, sim_image, index_map = get_simulation_maps('strebelle',
                                                           df.sample().iloc[0].at['simulation_parameters'])
with sns.axes_style('white'):
    fig, (ax1, ax2) = plt.subplots(1,2)
    p1 = ax1.imshow(sample.iloc[0].at['index_map'], cmap='viridis')
    ax1.axis('off')
    fig.colorbar(p1, ax=ax1, shrink=0.45)
    sample_intensity = verbatim_intensity(sample.iloc[0].at['index_map'])
    p2 = ax2.imshow(sample_intensity)
    fig.colorbar(p2, ax=ax2, shrink=0.45)
    ax2.axis('off')
    fig.savefig('img/noise_input_intensity_output.png', dpi=600, bbox_inches='tight')
    #plt.show()

 %%
