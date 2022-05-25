import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import disk
# perfect
# Fully random
from verbatim_metrics.data import get_simulation_maps, df
from verbatim_metrics.generate_data import generate_synthetic_data
from verbatim_metrics.local_statistics import verbatim_intensity

training_image, sim_image, index_map = get_simulation_maps('stone',
                                                           df.sample().iloc[0].at['simulation_parameters'])
maps = generate_synthetic_data()

#%% Index map and perfect map for explaining verbatim intensity
perfect_map = np.arange(200 ** 2).reshape((200, 200))
plt.subplot(121)
plt.title('Index origin matrix')
plt.imshow(index_map)
plt.subplot(122)
plt.title('Perfect matrix')
plt.imshow(perfect_map)
plt.show()
#%%
plt.title('Verbatim copy')
plt.imshow(verbatim_intensity(index_map,b=10))
plt.show()
#%% Noisy circles and verbatim intensity
plt.rcParams["figure.constrained_layout.use"] = True
plotloc = 330
plt.suptitle('Noisy circles synthetic data of varying verbatim')
plt.subplot(231)
plt.axis('off')
plt.title(maps.verbatim_percentage[1])
plt.imshow(maps.index_map[1])
plt.subplot(232)
plt.axis('off')
plt.title(maps.verbatim_percentage[100])
plt.imshow(maps.index_map[100])
plt.subplot(233)
plt.axis('off')
plt.title(maps.verbatim_percentage[800])
plt.imshow(maps.index_map[800])
plt.subplot(234)
plt.axis('off')
plt.title(np.round(np.mean(verbatim_intensity(maps.index_map[1],b=2)),5))
plt.imshow(verbatim_intensity(maps.index_map[1]))
plt.subplot(235)
plt.axis('off')
plt.title(np.round(np.mean(verbatim_intensity(maps.index_map[100],b=2)),5))
plt.imshow(verbatim_intensity(maps.index_map[100]))
plt.subplot(236)
plt.axis('off')
plt.title(np.round(np.mean(verbatim_intensity(maps.index_map[800],b=2)),5))
plt.imshow(verbatim_intensity(maps.index_map[800]))
plt.show()

#plt.savefig('img/noisy_circles.png')
#%%
verbatim = verbatim_intensity(random_map)
plt.imshow(verbatim)
plt.show()
