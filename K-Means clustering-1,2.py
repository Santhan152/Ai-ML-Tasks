#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### Clustering - Divide the universities in to groups (Clusters)

# In[3]:


Univ=pd.read_csv("Universities.csv")
Univ


# In[4]:


Univ.describe()


# In[5]:


Univ.info()


# In[6]:


Univ.isnull()


# In[7]:


Univ.sum()


# ### Standardization of the data

# In[9]:


Univ1=Univ.iloc[:,1:]
Univ1


# In[10]:


cols=Univ1.columns
cols


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1),columns=cols)
scaled_Univ_df


# In[12]:


from sklearn.cluster import KMeans
clusters_new=KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[13]:


clusters_new.labels_


# In[14]:


set(clusters_new.labels_)


# In[15]:


Univ['clusterid_new']=clusters_new.labels_


# In[16]:


Univ


# In[17]:


Univ.sort_values(by="clusterid_new")


# In[18]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations:
# - Cluster 1 appears to be on top rated universities cluster as the cut off score, Top10, SFratio parameter mean values are high.
# - Cluster 2 appears to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities.

# In[20]:


Univ[Univ['clusterid_new']==0]


# In[21]:


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

# In[23]:


from sklearn.cluster import KMeans
clusters_new=KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[44]:


from sklearn.metrics import silhouette_score
score=silhouette_score(scaled_Univ_df, clusters_new.labels_, metric='euclidean')
score


# In[46]:




