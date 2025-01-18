#!/usr/bin/env python
# coding: utf-8

# In[1]:


def remove_punc (input_text):
    punctuation_marks=['.',',','!','?','#','$','%','^','&','*']
    output_text=""
    for char in input_text:
        if char not in punctuation_marks:
            output_text +=char
    return output_text


# In[44]:


def remove_stopwords(input_text):
    stop_words=['an','is','why','and','how','where','%','on','of','whom']
    words=input_text.split()
    filtered_words=[]
    for word in words:
        if word.lower() not in stop_words:
            filtered_words.append(word)
    output_text=' '.join(filtered_words)
    return (output_text)
                         
            
            


# In[46]:


remove_stopwords('''why and how a lion is called as king of the forest''')


# In[ ]:




