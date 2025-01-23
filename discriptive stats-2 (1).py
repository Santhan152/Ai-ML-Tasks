#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("Universities.csv")
df


# In[55]:


np.mean(df["SAT"])


# In[4]:


np.median(df["SAT"])


# In[5]:


np.std(df["GradRate"])


# In[6]:


np.var(df["SFRatio"])


# In[7]:


df.describe()


# ### VISUALISATION

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df=pd.read_csv("Universities.csv")


# In[10]:


plt.hist(df["GradRate"])


# In[11]:


plt.figure(figsize=(6,3))


# In[12]:


plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# Visualization using boxplot

# In[33]:


s=[20,15,56,43,23,45,32]
scores=pd.Series(s)
scores


# In[38]:


plt.boxplot(scores,vert=True)


# In[49]:


s=[20,15,26,21,23,25,32,120,150]
scores=pd.Series(s)
scores


# In[51]:


plt.boxplot(scores,vert=False)


# Indentifying outliers from university dataset

# In[57]:


df=pd.read_csv("Universities.csv")
df


# In[65]:


plt.title("SAT")
plt.hist(df["SAT"])


# In[71]:


plt.boxplot(df["SAT"],vert=False)


# In[73]:


plt.boxplot(df["Top10"],vert=False)


# In[75]:


plt.boxplot(df["SFRatio"],vert=False)


# In[77]:


plt.boxplot(df["Expenses"],vert=False)


# In[ ]:




