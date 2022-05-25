#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from verbatim_metrics.data import df, get_simulation_maps
from verbatim_metrics.plot import plot_dendrogram

def pca_metrics():
    test = pattern_freq(index_map, b=10)
    freq = np.sum(test > 0, axis=0)
    freq
    plt.imshow(freq)
    plt.show()
    # %%
    X = test.reshape(test.shape[0], -1) > 0
    #        X = test.reshape(test.shape[0],-1)
    X[:, 220] = False

    pca = PCA(n_components=10)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    plt.title(np.round(pca.explained_variance_ratio_[0], 5))
    plt.imshow(pca.components_[0].reshape(21, 21))

# noinspection PyUnreachableCode
if False:
        #%% Testing
        training_image, sim_image, index_map = get_simulation_maps('stone',
                                                                   df.sample().iloc[0].at['simulation_parameters'])
        plt.imshow(index_map)
        plt.show()
        #%% Testing frequency windows
        test = pattern_freq(index_map, b=10)
        freq = np.sum(test > 0, axis=0)
        freq
        plt.imshow(freq)
        plt.show()
        #%%
        X = test.reshape(test.shape[0],-1) > 0
#        X = test.reshape(test.shape[0],-1)
        X[:,220] = False
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
        plt.show()
