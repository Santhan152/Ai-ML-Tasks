#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[60]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[62]:


data1.info()


# In[64]:


data1.describe()


# In[66]:


data1.isnull().sum()


# In[68]:


data1["daily"].corr(data1["sunday"])


# In[70]:


data1[["daily","sunday"]].corr()


# In[72]:


data1.corr(numeric_only=True)


# In[74]:


plt.scatter(data1["daily"], data1["sunday"])


# In[76]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[78]:


import seaborn as sns
sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[80]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[82]:


model.summary()


# In[ ]:




