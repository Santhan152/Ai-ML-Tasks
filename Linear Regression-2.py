#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[4]:


data1=pd.read_csv("NewspaperData.csv")
data1.head()


# In[6]:


data1.info()


# In[8]:


data1.isnull().sum()


# In[10]:


data1.describe()


# In[12]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"],vert=False)
plt.show()


# In[14]:


sns.histplot(data1["daily"],kde=True,stat='density',)
plt.show()


# In[16]:


sns.histplot(data1["sunday"],kde=True,stat='density',)
plt.show()


# In[18]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"],vert=False)
plt.show()


# In[20]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) +100)
plt.ylim(0, max(y) +100)
plt.show()


# In[22]:


data1["daily"].corr(data1["sunday"])


# In[24]:


data1[["daily","sunday"]].corr()


# In[26]:


data1.corr(numeric_only=True)


# In[28]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data=data1).fit()


# In[30]:


model1.summary()


# In[ ]:




