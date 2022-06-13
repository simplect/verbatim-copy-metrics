import numpy as np
import pandas as pd
from skimage.draw import disk


def generate_noisy_circles_map(mean_number_of_patches, map_size=200, noise=0.06, perfect_circles=True):
    max_index = map_size ** 2
    source_map = np.arange(max_index).reshape((map_size, map_size))
    random_map = np.random.randint(0,map_size**2,map_size**2).reshape((map_size,map_size))
    zero_map = np.full_like(random_map, -1)

    num_patches = int(np.random.normal(loc=mean_number_of_patches, scale=5, size=1)[0])
    num_patches = 0 if num_patches < 1 else num_patches

    patch_radii = np.random.normal(loc=2, scale=10, size=num_patches)
    # filter out negative
    patch_radii = patch_radii[patch_radii > 1]
    num_patches = len(patch_radii)
    for i in range(num_patches):
        patch_radius = patch_radii[i]
        target_location = np.random.randint(patch_radius, map_size-patch_radius, 2)
        source_location = np.random.randint(patch_radius, map_size-patch_radius, 2)

        rr1, cc1 = disk(target_location, patch_radius)
        rr2, cc2 = disk(source_location, patch_radius)
        # Dropout

        noise_mask = np.random.binomial(1, noise, len(rr1)) == (0 if perfect_circles else 1)

        rr1 = rr1[noise_mask]
        rr2 = rr2[noise_mask]
        cc1 = cc1[noise_mask]
        cc2 = cc2[noise_mask]

        random_map[rr1, cc1] = source_map[rr2, cc2]
        zero_map[rr1, cc1] = source_map[rr2, cc2]
    # Random map, percentage of pixels that have verbatim, number of patches, intensity
    return random_map, (np.sum(zero_map > -1)/max_index)*100, num_patches


def generate_synthetic_data(patches_range = 100, samples_per_param=10, noise=0.06):
    d = []
    for u_patches in np.arange(patches_range):
        for _ in range(samples_per_param):
            icm, truth, num_patches = generate_noisy_circles_map(u_patches, noise=noise)
            d.append({'u_patches': u_patches,
                      'index_map': icm,
                      'verbatim_percentage': truth,
                      'n_patches': num_patches})
    return pd.DataFrame(d)
