import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import disk
# perfect
# Fully random
from verbatim_metrics.data import get_simulation_maps, df
from verbatim_metrics.local_statistics import verbatim_intensity


def generate_noisy_circles_map(u_patches, l=200):
    l = 200
    max_index = l ** 2
    perfect_map = np.arange(max_index).reshape((l, l))
    random_map = np.random.randint(0,l**2,l**2).reshape((l,l))
    zero_map = np.full_like(random_map, -1)

    num_patches = int(np.random.normal(loc=u_patches, scale=5, size=1)[0])
    num_patches = 0 if num_patches < 1 else num_patches

    radii = np.random.normal(loc=2, scale=10, size=num_patches)
    # filter out negative
    radii = radii[radii > 1]
    num_patches = len(radii)
    for i in range(num_patches):
        radius = radii[i]
        random_place = np.random.randint(radius, l-radius, 2)
        perfect_place = np.random.randint(radius, l-radius, 2)

        rr1, cc1 = disk(random_place, radius)
        rr2, cc2 = disk(perfect_place, radius)
        noise_mask = np.random.binomial(1, 0.06, len(rr1)) == 0
        rr1 = rr1[noise_mask]
        rr2 = rr2[noise_mask]
        cc1 = cc1[noise_mask]
        cc2 = cc2[noise_mask]

        random_map[rr1, cc1] = perfect_map[rr2, cc2]
        zero_map[rr1, cc1] = perfect_map[rr2, cc2]
    return random_map, (np.sum(zero_map > -1)/max_index)


def generate_synthetic_data(patches_range = 100, samples_per_param=10):
    maps = []
    params = []
    verbatim_percentages = []
    for u_patches in np.arange(100):
        for _ in range(samples_per_param):
            map, truth = generate_noisy_circles_map(u_patches)
            verbatim_percentages.append(truth)
            params.append(u_patches)
            maps.append(map)

    return pd.DataFrame({'u_patches':params,
                         'index_map':maps,
                         'verbatim_percentage': verbatim_percentages})
