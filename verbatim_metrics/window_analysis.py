#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from verbatim_metrics.data import df_full, get_simulation_maps
from verbatim_metrics.local_statistics import pattern_freq
from verbatim_metrics.plot import plot_dendrogram
from verbatim_metrics.generate_data import generate_noisy_circles_map

def pca_metrics(index_map):
    test = pattern_freq(index_map, b=10)
    freq = np.sum(test > 0, axis=0)
    freq
    plt.imshow(freq)
    plt.show()
    X = test.reshape(test.shape[0], -1) > 0
    #        X = test.reshape(test.shape[0],-1)
    X[:, 220] = False

    pca = PCA(n_components=10)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    plt.title(np.round(pca.explained_variance_ratio_[0], 5))
    plt.imshow(pca.components_[0].reshape(21, 21))
# TODO: Can we calibrate PCA? Sample the windows instead of using all windows
# noinspection PyUnreachableCode
if False:
    param = df.sample().iloc[0].at['simulation_parameters']
    param
    #%% Testing
    samples = 100
    params = []
    ratios = []
    for param in df_full['simulation_parameters'].unique():
        ratios_sample = []
        for _ in range(samples):
            training_image, sim_image, index_map = get_simulation_maps('stone',
                                                                       param)
            test = pattern_freq(index_map, b=10)
            X = test.reshape(test.shape[0], -1) > 0
            X[:, 220] = False # Middle pixel zero
            pca = PCA(n_components=3)
            pca.fit(X)
            #print(pca.explained_variance_ratio_)
            ratios_sample.append(pca.explained_variance_ratio_)
            #plt.title(np.round(pca.explained_variance_ratio_[0], 5))
            #plt.imshow(pca.components_[0].reshape(21, 21))
        print(param)
        print(np.mean(np.array(ratios_sample), axis=0))
        params.append(param)
        ratios.append(np.mean(np.array(ratios_sample), axis=0))

        #plt.imshow(index_map)
        #plt.show()

        results = pd.DataFrame({'param': params, 'ratios': ratios})
        results.to_pickle('data/windows.pickle')

    #%%
    df['simulation_parameters'].unique()

    #%% Testing frequency windows
    index_map_n,verbatim = generate_noisy_circles_map(100)
    index_map, verbatim = generate_noisy_circles_map(10, perfect_circles=True, noise=0.6)
#    training_image, sim_image, index_map = get_simulation_maps('stone',
#                                                               df.sample().iloc[0].at['simulation_parameters'])
    #%%
    test = pattern_freq(index_map_n, b=10)
    X_n = test.reshape(test.shape[0],-1) > 0
    X_n[:,220] = False
    test = pattern_freq(index_map, b=10)
    #X = test.reshape(test.shape[0],-1) > 0
    #X[:,220] = False
    from sklearn.decomposition import PCA

    pca = PCA(n_components=10)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    #print(pca.singular_values_[0])
    plt.subplot(232)
    plt.imshow(index_map)
    plt.title('index origin')
    plt.axis('off')
    plt.subplot(234)
    plt.title(np.round(pca.explained_variance_ratio_[0],5))
    plt.imshow(pca.components_[0].reshape(21,21))
    plt.axis('off')
    plt.subplot(235)
    plt.title(np.round(pca.explained_variance_ratio_[1],5))
    plt.imshow(pca.components_[1].reshape(21,21))
    plt.axis('off')
    plt.subplot(236)
    plt.title(np.round(pca.explained_variance_ratio_[2],5))
    plt.imshow(pca.components_[2].reshape(21, 21))
    plt.axis('off')
    plt.suptitle(f'verbatim: {verbatim}')
    plt.show()

    fc = pca.transform(X)[:,0]
    plt.imshow(fc.reshape(200,200))
    plt.show()
