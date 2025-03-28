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

# In[7]:


cars.info()


# In[8]:


cars.isna().sum()


# ### Observations about info(), missing values
# - There are no missing value (81 different cars dsata)
# - The data types of the columns are also relevant and valid
# - There are 81 observations 

# In[10]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x="HP",ax=ax_box, orient="h")
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30, kde=True, stat="density")
ax_hist.set(ylabel="Density")
plt.tight_layout()
plt.show()


# In[11]:


ig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x="VOL",ax=ax_box, orient="h")
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins=30, kde=True, stat="density")
ax_hist.set(ylabel="Density")
plt.tight_layout()
plt.show()


# In[12]:


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

# In[14]:


ig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x="SP",ax=ax_box, orient="h")
ax_box.set(xlabel='')
sns.histplot(data=cars,x='SP',ax=ax_hist,bins=30, kde=True, stat="density")
ax_hist.set(ylabel="Density")
plt.tight_layout()
plt.show()


# ### Checking for duplicated rows

# In[16]:


cars[cars.duplicated()]


# ### Pair plots and correlation coefficients

# In[18]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# ### Observations from correlation plots and coefficients
# - Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG.
# - Therefore this dataset qualifies for building a multiple linear regression model to predict MPG.
# - Among x columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP, VOL vs WT.
# - The high correlation among x columns is not desirable as it might lead to multicollinearity problem.

# In[20]:


cars.corr()


# ### Preparing a preliminary model considering all X columns

# In[22]:


model1=smf.ols('MPG~WT+VOL+SP+HP', data = cars).fit()


# In[23]:


model1.summary()


# ### Observations from model summary
# - The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns.
# - The probability value with respect to F-statistic is close to zero, indicating that all or some of X columns are significant.
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored.

# ### Performance metrics for model1

# In[26]:


df1=pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[27]:


pred_y1=model1.predict(cars.iloc[:,0:4])
df1["pred_y1"]=pred_y1
df1.head()


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np
cars=pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# In[50]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# ### Checking for multicollinearity among X-columns using VIF method

# In[55]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)
rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 
rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 
rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 
# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ### Observations for VIF values:
# - The ideal range of VIF values shall be between 0 to 10. However slightly higher values can be tolerated.
# - As seen from the very high VIF values for VOL and WT, it is clear that they are prone to multicollinearity problem.
# - Hence it is decided to drop one of the columns (either VOL or WT) to overcome the multicolinearity.
# - It is decided to drop WT and retain VOL column in further models.

# In[58]:


cars1=cars.drop("WT",axis=1)
cars1.head()


# In[62]:


model2=smf.ols("MPG~HP+VOL+SP",data=cars1).fit()
model2.summary()


# In[64]:


df2=pd.DataFrame()
df2["actual_y2"]=cars["MPG"]
df2.head()


# In[66]:


pred_y2=model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"]=pred_y2
df2.head()


# In[70]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# In[ ]:




