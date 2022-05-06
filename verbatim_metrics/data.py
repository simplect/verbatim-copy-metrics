import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot parameters
plt.rcParams["figure.dpi"] = 320 # Dependent on screen
plt.rcParams["figure.constrained_layout.use"] = True

# Configuration
config_simulations_path = '/Users/merijn/stack/3. Studie/36._thesis/36.04 simulations/QS_Simulation4Verbatim'
config_training_images = ['stone', 'strebelle']
config_training_parameters = [0]

# Parse filenames
files = [file for file in os.listdir(config_simulations_path) if file[0:5] == 'qsSim']
file_parameters = np.array([file[6:][:-4].split('_') for file in files if file[0:5] == 'qsSim'])
file_parameters
df = pd.DataFrame({'training_image_type': file_parameters[:, 0],
                   'simulation_parameters': file_parameters[:, 1],
                   'index_map': np.NAN})
# Keep only stone or strebelle, as their index map sizes are different
df = df[df['training_image_type'] == 'stone']
# Create a random sample
df = df.sample(n=100, replace=True)

# %%
def get_simulation_maps(training_image_type, parameter):
    sim_result = np.load(f'{config_simulations_path}/qsSim_{training_image_type}_{parameter}.npz')
    # Per simulation there are many simulations, for now we sample one
    i = np.random.randint(sim_result['indexMap'].shape[0])
    _index_map = sim_result['indexMap'][i, :, :]
    _training_image = sim_result['ti']
    _sim_image = sim_result['sim'][i, :, :]
    return _training_image, _sim_image, _index_map


def get_index_map(training_image_type, parameter):
    _, _, _index_map = get_simulation_maps(training_image_type, parameter)
    return _index_map

# Collect the index maps for all the samples
df.loc[:, 'index_map'] = df.apply(lambda x: get_index_map(x['training_image_type'],
                                                       x['simulation_parameters']), axis=1)
df = df.reset_index(drop=True)

#%%
def plot_sim_index(training_image, sim_image, index_map):
    sourceIndex = np.stack(
        np.meshgrid(np.arange(training_image.shape[0]) / training_image.shape[0],
                    np.arange(training_image.shape[1]) / training_image.shape[1]) +
        [np.ones_like(training_image)],
        axis=-1);
    fig, axs = plt.subplots(2, 2)
    ax1 = plt.subplot(221)
    ax1.imshow(training_image)
    ax1.set_title('Training image')
    ax1.axis('off')
    ax2 = plt.subplot(222)
    ax2.imshow(sim_image)
    ax2.set_title('Simulation')
    ax2.axis('off')

    ax3 = plt.subplot(223)
    ax3.imshow(sourceIndex)
    ax3.set_title('Index origin')
    ax3.axis('off')
    ax3 = plt.subplot(224)
    ax3.imshow(index_map)
    ax3.imshow(sourceIndex.reshape(-1,3)[index_map])
    ax3.set_title('Index map')
    ax3.axis('off')
    plt.show()

if __name__=='__main__':
    training_image, sim_image, index_map = get_simulation_maps('stone', df.sample().iloc[0].at['simulation_parameters'])
    plot_sim_index(training_image, sim_image, index_map)
    #%%
    plt.hist(np.sort(index_map.reshape(-1)), bins=100)
    plt.show()


#%%
