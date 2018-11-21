
# coding: utf-8

# In[3]:


import skimage


# In[17]:


import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[8]:


from skimage import data
coins=data.coins()


# In[13]:


plt.imshow(coins,cmap='gray')


# In[26]:


from skimage import filters
coins_denoised=filters.median(coins,selem=np.ones((5,5)))
f,(ax0,ax1)=plt.subplots(1,2,figsize=(15,5))
ax0.imshow(coins)
ax1.imshow(coins_denoised)


# In[31]:


from skimage import feature
edges=skimage.feature.canny(coins,sigma=3)
plt.imshow(edges)


# In[32]:


from scipy.ndimage import distance_transform_edt
dt=distance_transform_edt(~edges)
plt.imshow(dt)


# In[36]:


local_max=feature.peak_local_max(dt,indices=False,min_distance=5)
plt.imshow(local_max,cmap='gray')


# In[38]:


peak_idx=feature.peak_local_max(dt,indices=True,min_distance=5)
peak_idx[:5]


# In[40]:


plt.plot(peak_idx[:,1],peak_idx[:,0],'r.')
plt.imshow(dt)


# In[41]:


from skimage import measure
markers=measure.label(local_max)


# In[50]:


from skimage import morphology,segmentation
labels=morphology.watershed(-dt,markers)
plt.imshow(segmentation.mark_boundaries(coins,labels))


# In[51]:


from skimage import color
plt.imshow(color.label2rgb(labels,image=coins))


# In[52]:


plt.imshow(color.label2rgb(labels,image=coins,kind='avg'),cmap='gray')


# In[53]:


regions=measure.regionprops(labels,intensity_image=coins)


# In[56]:


region_means=[r.mean_intensity for r in regions]
plt.hist(region_means,bins=20)


# In[61]:


from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
region_means=np.array(region_means).reshape(-1,1)


# In[67]:


model.fit(region_means)
print model.cluster_centers_


# In[68]:


bg_fg_labels=model.predict(region_means)
bg_fg_labels


# In[71]:


classified_labels=labels.copy()
for bg_fg,region in zip(bg_fg_labels,regions):
    classified_labels[tuple(region.coords.T)]=bg_fg


# In[72]:


plt.imshow(color.label2rgb(classified_labels,image=coins))

