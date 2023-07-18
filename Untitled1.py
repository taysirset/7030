#!/usr/bin/env python
# coding: utf-8

# ## Importing the dataset

# In[4]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# ## set notebook variables

# In[1]:


filename = "regrex1.csv"


# ## use read_csv() to read regrex1.csv file

# In[5]:


dataset = pd.read_csv(filename)
dataset.describe()
dataset


# ## Fitting linear regression to the Dataset

# In[6]:


model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# ## Visualize the linear regression results

# In[7]:


plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.xlabel('x of y')
plt.ylabel('y')
plt.show()


# In[8]:


## adjusted r-squared


# In[9]:


model.score(dataset[['x']], dataset[['y']])


# In[ ]:




