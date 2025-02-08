#!/usr/bin/env python
# coding: utf-8

# Assumption in Multilinear Regression
# 
#     1.Linearity:The relasrionship between the predictors(x) and the response(y) is linear
#     2.independence:Observations are independent of each other
#     3.Homoscedasticity:The residuals (y_y_hat) exhibit constant variance at all levels of the predictor
#     4.Normal Distribution of Errors:The residuals of the model are normally distrubuted

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars=pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars=pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# Description of columns
# 
#     * MPG:mileage of the car(mile per gallon)(This is y-column to be predicted)
#     * HP:Horse Power of the car(x1 column)
#     * VOL:volume of the car(size)(x2 column)
#     * SP:Top Speed of the car(MIles per HOur)(x3 column)
#     * WT:Weight of the car(pounds)(x4 column)

# In[4]:


cars.info()


# In[5]:


cars.isna().sum()


# Observations:
# 
#     * There are no missing values
#     * There are 81 different cars data
#     * The data types of the columns are relevent and valid

# In[6]:


fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[7]:


fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='VOL',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[8]:


fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='SP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='SP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[9]:


fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='WT',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='WT',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# Observation from boxplotand hitogrms
# 
#     *There are some extreme values obsereved in towards the right tail of sp and hp distributions
#     *In vol and wt columns,a few outliers are observed in both tails of distributions
#     *The extreme values of cars data may have come from the specially desgined nature of cars
#     *As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression model

# In[10]:


cars[cars.duplicated()]


# In[11]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[12]:


cars.corr()


# Observations from correlation plots and coeffcients
#    
#     * Between x and y,all the x variables are showing moderate to high correlation strengehs,highest being between HP and MPG
#     * Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
#     * Among x columns(x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP,VOL vs WT
#     * The high correlation among x columns is not desirable as it might lead to multicolinearity problem

# In[13]:


model1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[14]:


model1.summary()


# Observations from model summary
# 
#     * The R-squared and adjusted R-squared values are good and about 75% of variability is explained by x columns
#     * The probability value w.r.t to F-statistics is close to zero, indicating that all or some of x columns are significant
#     * The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored

# Performance metrics for model1

# In[15]:


df1=pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[16]:


pred_y1=model1.predict(cars.iloc[:,0:4])
df1["pred_y1"]=pred_y1
df1.head()


# In[17]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# In[18]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[19]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# In[20]:


rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[21]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[22]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[23]:


model2.summary()


# In[24]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[25]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[26]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# In[27]:


k=3
n=81
leverage_cutoff=3*((k+1)/n)
leverage_cutoff


# In[28]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=.05)
y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# ### Observations
# - From the above plot, it is evident that data points 65,70,76,78,79,80 are the influences.

# In[29]:


cars1[cars1.index.isin([65,70,76,78,79,80])]
cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# ### Build model3 on cars2 dataset

# In[30]:


model3=smf.ols('MPG~VOL+SP+HP',data=cars2).fit()


# In[31]:


model3.summary()


# ### Performance Metrics for model3

# In[32]:


df3=pd.DataFrame()
df3["actual_y3"]=cars2["MPG"]
df3.head()


# In[35]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[36]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[ ]:




