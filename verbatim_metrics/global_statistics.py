import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

from verbatim_metrics.data import df, get_simulation_maps

training_image, sim_image, index_map = get_simulation_maps('stone',
                                                           df.sample().iloc[0].at['simulation_parameters'])


#%%
def global_metric(index_map):
    metrics = {}
    X = np.unique(index_map, return_counts=True)[1]
    metrics['mean'] = np.mean(X)
    metrics['variance'] = np.var(X)
    metrics['median'] = np.median(X)

    metrics['pixels_used'] = np.sum(X > 1)
    metrics['pixels_power'] = np.sum(X > 1) * np.mean(X)
    metrics['pixels_reused'] = np.sum(X > 1) / len(X)
    return metrics

global_metric(index_map)

#%%
X = index_map.reshape(-1)
X = np.sort(np.unique(X, return_counts=True)[1])
plt.plot(X)
plt.show()
X

#%%
dicts_metrics = np.reshape(df['index_map'], -1).apply(global_metric)
#%%
for x in dicts_metrics.items():
    for y in x[1].items():
        df.loc[x[0], y[0]] = y[1]
#%%
# Index(['training_image_type', 'simulation_parameters', 'index_map', 'mean',
#        'variance', 'meanlog', 'variancelog', 'lincoef'],
test_var = 'pixels_reused'
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

