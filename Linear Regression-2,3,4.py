#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1=pd.read_csv("NewspaperData.csv")
data1.head()


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"],vert=False)
plt.show()


# In[7]:


sns.histplot(data1["daily"],kde=True,stat='density',)
plt.show()


# In[8]:


sns.histplot(data1["sunday"],kde=True,stat='density',)
plt.show()


# In[9]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"],vert=False)
plt.show()


# In[10]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) +100)
plt.ylim(0, max(y) +100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


data1[["daily","sunday"]].corr()


# In[13]:


data1.corr(numeric_only=True)


# In[14]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data=data1).fit()


# In[15]:


model1.summary()


# ### Interpretation:
# - R^2=1 --> Perfect fit (all variance explained).
# - R^2=0 --> Model does not explain any variance.
# - R^2 close to 1 --> Good model fit.
# - R^2 close to 0 --> Poor model fit.

# In[17]:


x=data1["daily"].values
y=data1["sunday"].values
plt.scatter(x,y,color="m",marker="o",s=30)
b0=13.84
b1=1.33
y_hat= b0 + b1*x
plt.plot(x,y_hat,color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ### Observations from model summary:
# - The probability (p-value) for intercept(beta_0) is 0.707 > 0.05.
# - Therefore the intercept coefficient may not be much significant in prediction.
# - However the p-value for "daily"(beta_1) is 0.00 < 0.05.
# - Therefore the beta_1 coefficient is highly significant and is contribute to prediction.
# - R^2 - value = 0.9.

# In[19]:


model1.params


# In[20]:


print(f'model t-values:\n{model1.tvalues}\n-------------\nmodel p-values: \n{model1.pvalues}')


# In[21]:


(model1.rsquared,model1.rsquared_adj)


# ### Predict for new data point

# In[23]:


newdata=pd.Series([200,300,1500])


# In[24]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[25]:


model1.predict(data_pred)


# In[26]:


pred= model1.predict(data1["daily"])
pred


# In[27]:


data1["Y_hat"]= pred
data1


# In[28]:


data1["residuals"]=data1["sunday"]-data1["Y_hat"]
data1


# In[57]:


mae=np.mean(np.abs(data1["daily"]-data1["Y_hat"]))
mae


# In[29]:


mse=np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse=np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# ### Checking the model residuals scatter plot (fOR homoscedasticity)

# In[62]:


plt.scatter(data1["Y_hat"],data1["residuals"])


# ### Observations:
# - There appears to be no trend and the residuals are randomly placed around the zero error line.
# - Hence the assumption of homoscedasticity (constant variance in residuals) is satisfied.

# In[67]:


import statsmodels.api as sm
sm.qqplot(data1["residuals"],line='45',fit=True)
plt.show()


# In[69]:


sns.histplot(data1["residuals"],kde=True)


# ### Observations:
# - The data points are seen to closely follow the refrence line of normality
# - Hence the residuals are approximately normally distributed as also can be seen from the kde distribution.

# ### Model Improvements:

# In[ ]:




