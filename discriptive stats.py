#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[8]:


df=pd.read_csv("Universities.csv")
df


# In[10]:


np.mean(df["SAT"])


# In[12]:


np.median(df["SAT"])


# In[16]:


np.std(df["GradRate"])


# In[18]:


np.var(df["SFRatio"])


# In[20]:


df.describe()


# In[ ]:




