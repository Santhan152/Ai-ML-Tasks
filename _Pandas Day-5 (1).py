#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas
import pandas as pd


# In[3]:


#create pandas series using list
data=[10,20,30,40]
series=pd.Series(data)
print(series)


# In[5]:


#create pandas series using a custom index value
data=[1,2,3,4]
i=['A','B','C','D']
series=pd.Series(data, index=i)
print(series)


# In[7]:


#create pandas series using dictionary
data={'a':10,'b':20,'c':30,'d':40}
series=pd.Series(data)
print(series)


# In[9]:


series.replace(20,100)


# In[11]:


#create series using numpy array


# Pandas dataframe

# In[17]:


#create pandas dataframe from dictionary of lists


# In[19]:


import pandas as pd


# In[23]:


data={'Name':['Alice','Bob','Mary'],'Age':[25,30,34],'Country':['Usa','India','Russia']}
df=pd.DataFrame(data)
print(df)


# In[25]:


#create pandas dataframe from numpy array


# In[31]:


import numpy as np
array=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array)
df=pd.DataFrame(array,columns=['A','B','C'])
print(df)


# In[9]:


#create a dataframe from a nested list
import pandas as pd
data=[[1,'Alice',25],[2,'Bob',30],[3,'Mary',34]]
print(data)
df=pd.DataFrame(data, columns=['Id','Name','Age'])
print(df)


# In[40]:


#import data from a csv file and create
iris_df=pd.read_csv("iris.csv")
print(iris_df)


# In[15]:


import pandas as pd
data=[['2311cs020152','Bhagu',1,7.8],['2311cs020152','Jai',2,8.9],['2311cs020152','Santhan',3,8.1],['2311cs020152','Navaneeth',4,9.9]]
print(data)
df=pd.DataFrame(data, columns=['Roll number','Name','semmester','SGPA'])
print(df)


# In[21]:


#import data from a csv file and create
iris_df=pd.read_csv("iris.csv")
print(iris_df)


# In[23]:


iris_df.info()


# In[25]:


iris_df.head()


# In[27]:


iris_df.tail()


# In[31]:


iris_df.head(150)


# In[33]:


iris_df.describe()


# In[35]:


iris_df.count()


# In[37]:


iris_df.mean()


# In[39]:


iris_df.std()


# In[41]:


print(iris_df.shape)
print(iris_df.ndim)
print(iris_df.size)


# In[45]:


import numpy as np
iris_array=np.array(iris_df)
iris_array


# In[53]:


iris_df.iloc[:,2]


# In[49]:


iris_df


# In[59]:


iris_df[["sepal.length","petal.width"]]


# In[75]:


iris_df.iloc[15:21,[0,3]]


# In[81]:


iris_df.loc[10:20:15,"sepal.length":"petal.length"]


# In[91]:


data={"weight":[66,75,84,96,48,71], 
      "height":[156,165,186,165,174,167]}
bmi=pd.DataFrame(data)
bmi


# In[93]:


#Create new BMI column
bmi["BMI"]=bmi["weight"]/(bmi["height"]/100)**2
bmi


# In[101]:


bmi.loc[3]=[78,np.nan,np.nan]
bmi


# In[103]:


bmi.loc[5]=[np.nan,65,np.nan]
bmi


# In[160]:


#create column wise size of
bmi["weight"]=bmi["weight"].fillna(165)
bmi


# In[166]:


bmi.drop("weight",axis=1,inplace=True)
bmi


# In[172]:


import matplotlib.pyplot as plt


# In[ ]:




