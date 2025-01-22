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


# ### VISUALISATION

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df=pd.read_csv("Universities.csv")


# In[14]:


plt.hist(df["GradRate"])


# In[20]:


plt.figure(figsize=(6,3))


# In[22]:


plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[ ]:




