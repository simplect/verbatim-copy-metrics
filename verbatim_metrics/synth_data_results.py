import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from verbatim_metrics.generate_data import generate_synthetic_data
from verbatim_metrics.global_statistics import global_metric
from verbatim_metrics.local_statistics import cluster_metric

df = generate_synthetic_data(patches_range=100, samples_per_param=10)

 # %%

#%%
def apply_metric(df, metric = cluster_metric):
    print(f'starting thread at {datetime.now().strftime("%H:%M")}')
    dicts_metrics = np.reshape(df['index_map'], -1).apply(metric)
    for x in dicts_metrics.items():
        for y in x[1].items():
            df.loc[x[0], y[0]] = y[1]
    return df
#df = apply_metric(df, global_metric)
#df = apply_metric(df, cluster_metric)
# %%
from joblib import Parallel, delayed
import multiprocessing

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=2, prefer='threads')(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)
print('Starting thread')
%time result = applyParallel(df.groupby(df.u_patches), apply_metric)
result.to_pickle('data/synth_cluster.pickle')
#%%
# Index(['training_image_type', 'simulation_parameters', 'index_map', 'mean',
#        'variance', 'meanlog', 'variancelog', 'lincoef'],
test_var = 'variance'
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
