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


# In[12]:


median_ozone = data1["Solar"].median()
mean_ozone=data1["Solar"].mean()
print("Median of Solar: ", median_ozone)
print("Mean of Solar: ", mean_ozone)


# In[13]:


print(data1["Weather"].value_counts())
mode_weather=data1["Weather"].mode()[0]
print(mode_weather)


# In[14]:


data1["Weather"]=data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[ ]:





# In[15]:


print(data1["Month"].value_counts())
mode_month=data1["Month"].mode()[0]
print(mode_month)


# In[16]:


data1["Month"]=data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[17]:


print(data1["Day"].value_counts())
mode_day=data1["Day"].mode()[0]
print(mode_day)


# In[18]:


data1["Day"]=data1["Day"].fillna(mode_day)
data1.isnull().sum()


# In[19]:


print(data1["Year"].value_counts())
mode_year=data1["Year"].mode()[0]
print(mode_year)


# In[20]:


data1["Year"]=data1["Year"].fillna(mode_year)
data1.isnull().sum()


# In[21]:


print(data1["Solar"].value_counts())
median_solar=data1["Solar"].median()
print(median_solar)


# In[22]:


data1["Solar"]=data1["Solar"].fillna(mode_solar)
data1.isnull().sum()


# In[ ]:


print(data1["Month"].value_counts())
mode_month=data1["Month"].mode()[0]
print(mode_month)


# In[ ]:


data1["Month"]=data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[ ]:


print(data1["Ozone"].value_counts())
mode_ozone=data1["Ozone"].mode()[0]
print(mode_ozone)


# In[ ]:


data1["Ozone"]=data1["Ozone"].fillna(mode_ozone)
data1.isnull().sum()


# Detection of outliers in the columns

# Method1: Using histograms and box plots

# In[ ]:


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

# In[ ]:


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


# In[ ]:


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


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"],vert=False)


# In[ ]:


plt.figure(figsize=(6,2))
boxplot_data=plt.boxplot(data1["Ozone"],vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# ### Method 2
# - **Using +/-3*sigma limits(Standard deviation method)** 

# In[ ]:


data1["Ozone"].describe()


# In[ ]:


mu=data1["Ozone"].describe()[1]
sigma=data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x<(mu - 3*sigma)) or (x>(mu+3*sigma))):
        print(x)
    


# Observations:
# It is observed that only two outliers are idenfied using std method
# In box plot method more no of outliers are identified
# This is because the asumption of normality is not satisfied in this column

# In[ ]:


import scipy.stats as stats
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"],dist="norm",plot=plt)
plt.title("Q-Q Plot for Outlier Detection",fontsize=14)
plt.xlabel("Theoretical Quantities",fontsize=12)


# In[ ]:


sns.violinplot(data=data1["Ozone"],color='lightgreen')
plt.title("Violin Plot")


# In[ ]:


mu=data1["Solar"].describe()[1]
sigma=data1["Solar"].describe()[2]
for x in data1["Solar"]:
    if ((x<(mu - 3*sigma)) or (x>(mu+3*sigma))):
        print(x)


# In[ ]:


import scipy.stats as stats
plt.figure(figsize=(8,6))
stats.probplot(data1["Solar"],dist="norm",plot=plt)
plt.title("Q-Q Plot for Outlier Detection",fontsize=14)
plt.xlabel("Theoretical Quantities",fontsize=12)


# In[ ]:


sns.violinplot(data=data1["Solar"],color='lightgreen')
plt.title("Violin Plot")


# Other visualisations 

# In[ ]:


sns.violinplot(data=data1["Ozone"],color='lightgreen')
plt.title("Violin Plot")
plt.show()


# In[36]:


sns.swarmplot(data=data1,x="Weather",y="Ozone",color="Orange",palette="Set1",size=6)


# In[32]:


sns.stripplot(data=data1,x="Weather",y="Ozone",color="Orange",palette="Set2",size=6,jitter=True)


# In[30]:


sns.kdeplot(data=data1["Ozone"],fill=True,color="blue")
sns.rugplot(data=data1["Ozone"],color="black")


# In[38]:


sns.boxplot(data=data1,x="Weather",y="Ozone")


# Correlation coefficient and pair plots

# In[40]:


plt.scatter(data["Wind"],data1["Temp"])


# In[45]:


data1["Wind"].corr(data1["Temp"])


# In[51]:


data1_numeric=data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[53]:


plt.scatter(data["Wind"],data1["Ozone"])
plt.scatter(data["Ozone"],data1["Solar"])
plt.scatter(data["Wind"],data1["Temp"])


# In[ ]:




