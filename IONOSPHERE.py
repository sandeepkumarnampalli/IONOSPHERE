#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("c:/users/sande/Ionosphere.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# # Finding that whether this is a Regression or Classification Model by EDA.

# In[5]:


df["Class"].value_counts()


# In[6]:


df["Class"].value_counts().plot()
plt.show()


# In[7]:


# Inference 1: By looking at the graph it is representing finite values so,this is classificaton model.
# Inference 2: This is logistics Regression model used for Classification model having finite variables.


# In[8]:


sns.countplot("Class",data=df)
plt.show()


# In[9]:


# Inference 3:  This is the Classification model have discrete variables.
# Inference 4:  Class 0 has no variance and therefore it is not useful to the model.


# In[10]:


y=df["Class"]


# In[11]:


X=df.drop("Class",axis=1)


# In[12]:


X.head()


# HERE WE HAVE TO FIRST FIND THE NAN VALUES IN THE DATASET.

# In[13]:


df.isna().sum()


# In[14]:


df.dtypes


# In[15]:


df.info()


# In[16]:


type(df)


# In[17]:


df.shape


# MACHINE LEARNING STEPS AND RULES TO BUILD A MODEL.
# 1. Extract features.<br/>
# a. Features and target should not have null values.<br/>
# b. Features should be of numeric in nature.<br/>
# c. Features should be of the type array/dataframe.<br/>
# d. Features should have some rows and columns.<br/>
# 2. Split the datasets into training datasets and testing datasets.<br/>
# e. Features should be on same scale.<br/>
# 3. Train the model on training datasets.<br/>
# 4. Testing the model on testing datasets.<br/>

# # 1.Extract features.

# # a.Features and target should not have null values.

# In[18]:


X.isna().sum()


# In[19]:


y.isna().sum()


# # b.Features should be of numeric in nature.

# In[20]:


X.dtypes


# # c.Features should be of the type array/dataframe.

# In[21]:


type(X)


# # d.Features should have some rows and columns.

# In[22]:


X.shape


# # 2.Split the datasets into training datasets and testing datasets.

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


# In[25]:


from sklearn.preprocessing import MinMaxScaler


# In[26]:


scaler=MinMaxScaler()


# In[27]:


X_train=scaler.fit_transform(X_train)


# In[28]:


X_test=scaler.transform(X_test)


# # 3.Train the model on training datasets.

# In[29]:


from sklearn.linear_model import LogisticRegression


# In[30]:


model=LogisticRegression()


# In[31]:


model.fit(X_train,y_train)


# # 4.Testing the model on testing datasets.

# In[32]:


model.score(X_test,y_test)


# In[33]:


model.score(X_train,y_train)


# In[34]:


# Here from the above Train score there is no overfitting in the Model.


# In[ ]:




