import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 320
plt.rcParams["figure.constrained_layout.use"] = True
i = 2
index_map = np.load(f'../sims/{i}/indexMap.npy')
simulation = np.load(f'../sims/{i}/simulation.npy')
source_index = np.load(f'../sims/{i}/sourceIndex.npy')
ti = np.load(f'../sims/{i}/ti.npy')

# In[2]:


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Unconditional simulation')
ax1.imshow(ti)
ax1.set_title('Training image')
ax1.axis('off')
ax2.imshow(simulation)
ax2.set_title('Simulation')
ax2.axis('off')
plt.autoscale(tight=True)
plt.show()

# In[58]:


sourceIndex = np.stack(
    np.meshgrid(np.arange(ti.shape[0]) / ti.shape[0], np.arange(ti.shape[1]) / ti.shape[1]) +\
    [np.ones_like(ti)],
    axis=-1);
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Unconditional simulation index map')
ax1.imshow(sourceIndex)
ax1.set_title('Training image');
ax1.axis('off');
ax2.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
#ax2.imshow(index_map)
ax2.set_title('Simulation');
ax2.axis('off');
plt.show()

# In[5]:


import imageio

# In[66]:


img = np.reshape(sourceIndex, (-1, 3))[index_map].shape
plt.imshow(index_map / 4e4)

# In[48]:


randimg = np.random.uniform(0, 4e4, 200 * 200).reshape(200, 200)
plt.imshow(randimg)

# In[76]:


import numpy as np
from skimage.feature import greycomatrix

# In[99]:


seqimg = np.arange(0, 4e4).reshape(200, 200)

# In[114]:


np.arange(0, 10)
glcm = np.squeeze(greycomatrix(np.uint8(index_map.T / 4e4 * 255), distances=[1],
                               angles=[0], symmetric=True,
                               normed=True))
glcm_rnd = np.squeeze(greycomatrix(np.uint8(randimg / 4e4 * 255), distances=[1],
                                   angles=[0], symmetric=True,
                                   normed=True))

glcm_seq = np.squeeze(greycomatrix(np.uint8(seqimg / 4e4 * 255), distances=[1],
                                   angles=[0], symmetric=True,
                                   normed=True))
glcm.shape, glcm_seq.shape

# In[115]:


plt.imshow(glcm)
plt.show()

# In[106]:


plt.imshow(glcm_seq)
plt.show()

# In[107]:


glcm_seq

# In[108]:


plt.imshow(glcm_rnd)
plt.show()

# In[90]:


np.matrix.diagonal(glcm)

# In[92]:


np.matrix.diagonal(glcm_rnd)

# In[96]:


entropy_seq = -np.sum(glcm_seq * np.log2(glcm_seq + (glcm_seq == 0)))
entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
# yields 10.704625483788325
entropy_rnd = -np.sum(glcm_rnd * np.log2(glcm_rnd + (glcm_rnd == 0)))
# yields 10.704625483788325
entropy, entropy_rnd, entropy_seq

# In[ ]:


# %%

# %%
