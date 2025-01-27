#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


# Drop duplicate column( Temp C) and Unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis = 1)
data1


# In[5]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[6]:


data1.rename({'Solar.R':'Solar'}, axis=1, inplace=True)
data1


# In[7]:


data1.info()


# In[8]:


data1.isnull().sum()


# In[9]:


cols=data1.columns
colors=['black','yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[10]:


median_ozone = data1["Ozone"].median()
mean_ozone=data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[11]:


data1['Ozone']=data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[13]:


median_ozone = data1["Solar"].median()
mean_ozone=data1["Solar"].mean()
print("Median of Solar: ", median_ozone)
print("Mean of Solar: ", mean_ozone)


# In[28]:


print(data1["Weather"].value_counts())
mode_weather=data1["Weather"].mode()[0]
print(mode_weather)


# In[30]:


data1["Weather"]=data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[ ]:





# In[32]:


print(data1["Month"].value_counts())
mode_month=data1["Month"].mode()[0]
print(mode_month)


# In[58]:


data1["Month"]=data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[34]:


print(data1["Day"].value_counts())
mode_day=data1["Day"].mode()[0]
print(mode_day)


# In[60]:


data1["Day"]=data1["Day"].fillna(mode_day)
data1.isnull().sum()


# In[36]:


print(data1["Year"].value_counts())
mode_year=data1["Year"].mode()[0]
print(mode_year)


# In[62]:


data1["Year"]=data1["Year"].fillna(mode_year)
data1.isnull().sum()


# In[52]:


print(data1["Solar"].value_counts())
median_solar=data1["Solar"].median()
print(median_solar)


# In[64]:


data1["Solar"]=data1["Solar"].fillna(mode_solar)
data1.isnull().sum()


# In[54]:


print(data1["Month"].value_counts())
mode_month=data1["Month"].mode()[0]
print(mode_month)


# In[66]:


data1["Month"]=data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[56]:


print(data1["Ozone"].value_counts())
mode_ozone=data1["Ozone"].mode()[0]
print(mode_ozone)


# In[68]:


data1["Ozone"]=data1["Ozone"].fillna(mode_ozone)
data1.isnull().sum()


# Detection of outliers in the columns

# Method1: Using histograms and box plots

# In[82]:


fig,axes=plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Ozone"],ax=axes[0],color="skyblue",width=0.5,orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data["Ozone"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# Observations
# *The ozone column has extreme values beyond 81 as seen from box plot
# *THe same is confirmed from the below right-skewed histogram

# In[100]:


fig,axes=plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Solar"],ax=axes[0],color="skyblue",width=0.5,orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")
sns.histplot(data1["Solar"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# In[96]:


sns.violinplot(data=data1["Solar"],color='pink')
axes[0].set_title("Violinplot")
axes[0].set_xlabel("Solar Levels")
sns.histplot(data1["Solar"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# In[ ]:




