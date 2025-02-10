#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### Clustering - Divide the universities in to groups (Clusters)

# In[10]:


Univ=pd.read_csv("Universities.csv")
Univ


# In[17]:


Univ.describe()


# In[21]:


Univ.info()


# In[32]:


Univ.isnull()


# In[36]:


Univ.sum()


# In[65]:


Univ1=Univ.iloc[:,1:]
Univ1


# In[69]:


cols=Univ1.columns
cols


# In[71]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1),columns=cols)
scaled_Univ_df


# In[ ]:




