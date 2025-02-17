#!/usr/bin/env python
# coding: utf-8

# In[18]:


###!pip install mlxtend


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules


# In[22]:


titanic=pd.read_csv("Titanic.csv")
titanic


# ### Observations:
# - There are no null values present in the dataset.
# - Almost no child has survived while almost all adults survived.
# - There are more childs in male and more adults in female.
# - All obejects are categorical in nature.
# - As the columns are categorical, we can adopt one-hot encoding.

# In[24]:


titanic.info()


# In[26]:


titanic.isna().sum()


# In[28]:


titanic.describe()


# In[40]:


counts=titanic['Class'].value_counts()
plt.bar(counts.index,counts.values)


# ### Observations:
# - The crew has travelled more than others.
# - The 2nd highest is the 3rd that has travelled more than others .

# In[54]:


titanic['Age'].value_counts()


# In[56]:


titanic['Gender'].value_counts()


# In[59]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# ### Observations:
# - So many columns are being created based on each category.
# - The table clearly shows or describes the data .

# In[62]:


df.info()


# ### Apriori Algorithm

# In[65]:


frequent_itemsets=apriori(df, min_support=0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[73]:


frequent_itemsets.iloc[62,1]


# In[75]:


frequent_itemsets.info()


# In[79]:


### Generate association rules with metrics:
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1.0)
rules


# ### Observations:
# - if (Gender_Female) then (Class_1st) and if (Gender_Female) then (Class_1st) has the highest lift.
# - Since all the lift values are more than 1 ,the table contains a stong and mild association

# In[84]:


rules.sort_values(by='lift',ascending =True)


# - As we move down the table the lift value is being increasing whic means as we move down the table the association is becoming more stronger.

# In[95]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




