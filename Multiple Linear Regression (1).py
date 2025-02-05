#!/usr/bin/env python
# coding: utf-8

# ### Assumptions:
#  1. Linearity: The relationship between the predictions(x) and the response (y) is linear.
#  2. Independance: Observations are independent of each other.
#  3. Homoscedasticity: The residuals (Y-Y_hat) exhibit constant variance at all levels of the predictor.
#  4. Normal Distribution of Errors: The residuals of the model are normally distributed.
#  5. No multicollinearity: The independent variables should not be too highly correlated with each other.
# ### Violation of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


cars=pd.read_csv("Cars.csv")
cars.head()


# In[4]:


cars=pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ### Descriptions of columns
# - MPG : Milege of the car(Mile per Gallon)
# - HP  : Horse Power of the car (x1 column)
# - VOL : Volume of the car (Size) (x2 column)
# - SP  : Top speed of the car (Miles per hour) (x3 column)
# - WT  : Weight of the car (Pounds) (x4 column)

# ### EDA

# In[9]:


cars.info()


# In[14]:


cars.isna().sum()


# ### Observations about info(), missing values
# - There are no missing value (81 different cars dsata)
# - The data types of the columns are also relevant and valid
# - There are 81 observations 

# In[18]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x="HP",ax=ax_box, orient="h")
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30, kde=True, stat="density")
ax_hist.set(ylabel="Density")
plt.tight_layout()
plt.show()


# In[22]:


ig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x="VOL",ax=ax_box, orient="h")
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins=30, kde=True, stat="density")
ax_hist.set(ylabel="Density")
plt.tight_layout()
plt.show()


# In[24]:


ig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x="WT",ax=ax_box, orient="h")
ax_box.set(xlabel='')
sns.histplot(data=cars,x='WT',ax=ax_hist,bins=30, kde=True, stat="density")
ax_hist.set(ylabel="Density")
plt.tight_layout()
plt.show()


# ### Observations from boxplot and histograms
# - There are some extreme values (outliers) observed in towards the right tail of SP and HP distributions.
# - In VOL andWT columns, a few outliers are observed in both tails of their distributions.
# - The extreme values of cars data may have come from the specially designed nature of cars.
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression model.

# In[26]:


ig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x="SP",ax=ax_box, orient="h")
ax_box.set(xlabel='')
sns.histplot(data=cars,x='SP',ax=ax_hist,bins=30, kde=True, stat="density")
ax_hist.set(ylabel="Density")
plt.tight_layout()
plt.show()


# ### Checking for duplicated rows

# In[35]:


cars[cars.duplicated()]


# ### Pair plots and correlation coefficients

# In[38]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[ ]:




