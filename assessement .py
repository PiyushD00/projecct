#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np 
import pandas as pd 
import load_heart()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, \
AdaBoostClassifier, \
GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, \
confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


df = pd.read_csv('heart.csv')
df


# In[ ]:





# In[39]:


x = heart['data']
y = heart['target']

target_names = heart['target_names']

print(x.shape)
print(y.shape)

print(target_names)


# In[28]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[29]:


model = RandomForestClassifier()
model.fit(x_train,y_train)


# In[30]:


y_pred = model.predict(x_test)
print(y_pred)


# In[31]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[32]:


model.score(x_test,y_test)


# In[33]:


print(accuracy_score(y_test,y_pred))


# In[40]:


print(classification_report(y_test,y_pred))


# In[41]:


df.shape


# In[42]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[43]:


df = pd.read_csv('heart.csv')
male_count = (df['sex'] == 1).sum()
female_count = (df['sex'] == 0).sum()
print(f'Number of males:{male_count}')
print(f'Number of females:{female_count}')


# In[44]:


df = pd.read_csv('heart.csv')
columns_of_interest = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'ca']
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
for i, column in enumerate(columns_of_interest):
    row = i // 3
    col = i % 3
    sns.histplot(df[column], ax=axes[row, col], kde=True)
    axes[row, col].set_title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# In[45]:


bins = [30,40,50,60,70,80]
labels = ['30-40','40-50','50-60','60-70','70-80']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

age_group = df['age_group'].value_counts().sort_index()
print("total patients of each age group",age_group)

common = age_group.idxmax()
print("most common age group is: ",common)


# In[ ]:




