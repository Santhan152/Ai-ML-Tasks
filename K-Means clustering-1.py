#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### Clustering - Divide the universities in to groups (Clusters)

# In[15]:


Univ=pd.read_csv("Universities.csv")
Univ


# In[17]:


Univ.describe()


# In[19]:


Univ.info()


# In[21]:


Univ.isnull()


# In[23]:


Univ.sum()


# ### Standardization of the data

# In[26]:


Univ1=Univ.iloc[:,1:]
Univ1


# In[28]:


cols=Univ1.columns
cols


# In[30]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1),columns=cols)
scaled_Univ_df


# In[32]:


from sklearn.cluster import KMeans
clusters_new=KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[35]:


clusters_new.labels_


# In[37]:


set(clusters_new.labels_)


# In[39]:


Univ['clusterid_new']=clusters_new.labels_


# In[41]:


Univ


# In[43]:


Univ.sort_values(by="clusterid_new")


# In[45]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations:
# - Cluster 1 appears to be on top rated universities cluster as the cut off score, Top10, SFratio parameter mean values are high.
# - Cluster 2 appears to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities.

# In[51]:


Univ[Univ['clusterid_new']==0]


# In[53]:


wcss=[]
for i in range(1,20):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,20),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ### Observations:
# - from the above graph we can choose k=3 or 4 which indicates the Elbow join ie; the rate of change of slope decreases.

# In[55]:


from sklearn.cluster import KMeans
clusters_new=KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[ ]:




