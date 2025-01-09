#!/usr/bin/env python
# coding: utf-8

# In[1]:


#conventional if-else


# In[3]:


n=6
if n%2==0:
    print("even")
else:
    print("odd")


# In[13]:


x=int(input("Enter the integer: "))
result ="Possitive" if x> 0 else "Negative"
print(result)


# In[11]:


age = int(input("Enter the age: "))
cateogory="Adult" if age>=18 else "Minor"
print(cateogory)


# In[19]:


number=int(input("Enter the integer: "))
result ="Possitive" if number> 0 else "Negative" if number<0 else "zero"
print(result)


# In[23]:


L=[100,1999999999999999,6,8]
x=int(input("Enter the value: "))
[2*x for x in L]


# In[25]:


L=[100,1999999999999999,6,8]
x=int(input("Enter the value: "))
[2+x for x in L]


# In[27]:


L=[100,1999999999999999,6,8]
x=int(input("Enter the value: "))
[2/x for x in L]


# In[29]:


L=[100,1999999999999999,6,8]
x=int(input("Enter the value: "))
[2%x for x in L]


# In[31]:


[x for x in L if x%2 !=0]


# In[33]:


[x for x in L if x%2 == 0]


# In[39]:


eyu={'ram':[2,3,4,5],'john':[6,7,8,9],'prem':[10,11,12,13]}


# In[41]:


{k:sum(v)/len(v) for k,v in eyu.items()}


# In[47]:


def mean_value(given_list):
    total=sum(given_list)
    average_value=total/len(given_list)
    return average_value


# In[49]:


#call the function
l=[1,23,45,67,89,35]
mean_value(l)


# In[1]:


def greet(Santhan):
    print(f"Good morning,{Santhan}!")
greet("Santhan")


# In[7]:


def avg_value(*n):
    l = len(n)
    average = sum(n)/l
    return average


# In[11]:


avg_value(10,20,60,100,900,1000)


# In[ ]:


#Lambda Functions


# In[25]:


greet = lambda name: print(f"Good Morning {name}!")


# In[27]:


greet("Santhan")


# In[29]:


#product of 3 numbers


# In[31]:


product = lambda a,b,c : a*b*c


# In[33]:


product(20,30,40)


# In[35]:


#lambda function with list comprehension


# In[37]:


even = lambda L : [x for x in L if x%2==0]


# In[43]:


my_list = [100,3,9,38,43,56,28]
even(my_list)


# In[49]:


def mean_value(given_list):
    total=sum(given_list)
    average_value=total/len(given_list)
    return average_value
l=[1,23,45,67,89,35]
mean_value(l)


# In[59]:


def median_value(*n):
    num_list = list(n)
    num_list.sort()
    l = len(num_list)
    if l%2 == 0:
        median = (num_list[int(l/2)]+num_list[int(l/2)-1])/2
    else:
        median = (num_list[int(l/2)])
    return median


# In[61]:


median_value(1,2,3,4,5,6,7,8,9,10)


# In[ ]:




