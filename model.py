#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle


# In[18]:


data = pd.read_csv(r"C:\Users\rameshwari.r.thorat\Documents\Bank_Marketing\bank-additional.csv",header='infer',sep=';')
data


# In[20]:


categorical_columns = data.select_dtypes(include=['category', 'object']).columns.tolist()
categorical_columns


# In[3]:


#data_new = pd.get_dummies(data, columns=['job','marital',
                                        # 'education','default',
                                         #'housing','loan',
                                         #'contact','month',
                                         #'poutcome','day_of_week'])


# In[21]:


le = LabelEncoder()
for i in categorical_columns:
    data[i] = le.fit_transform(data[i])


# In[23]:


data.head(5)


# In[24]:


data_y = pd.DataFrame(data['y'])
data_X = data.drop(['y'], axis=1)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=2, stratify=data_y)


# In[26]:


model_LR = LogisticRegression()
model_LR


# In[27]:


model_LR.fit(X_train, y_train)


# In[30]:


y_pred = model_LR.predict(X_test)

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[31]:


pickle.dump(model_LR, open('model.pkl','wb'))


# In[32]:


model = pickle.load(open('model.pkl','rb'))


# In[ ]:




