#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df_movies=pd.read_csv('movies.csv',sep="::",engine='python')


# In[5]:


df_movies


# In[6]:


df_users=pd.read_csv('users.csv',sep="::",engine='python')


# In[7]:


df_users


# In[9]:


df_movies.columns =['MovieIDs','MovieName','Category']


# In[10]:


df_movies


# In[11]:


df_movies.isna().sum()


# In[13]:


df_rating = pd.read_csv("ratings.csv",sep='::', engine='python')
df_rating.columns =['ID','MovieID','Ratings','TimeStamp']
df_rating.dropna(inplace=True)
df_rating.head()


# In[15]:


df_rating


# In[19]:


df_user=pd.read_csv('users.csv',sep="::",engine='python')
df_user.columns =['UserID','Gender','Age','Occupation','Zip-code']
df_user.isna().sum()
df_user.head()


# In[20]:


df = pd.concat([df_movies, df_rating,df_user], axis=1)
df.head()


# # 2. Perform the Exploratory Data Analysis (EDA) for the users dataset
# 

# In[22]:


#Visualize user age distribution
df['Age'].value_counts().plot(kind='barh',alpha=0.7,figsize=(10,10))
plt.show()


# In[23]:


df.Age.plot.hist(bins=25)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('Age')


# In[24]:


labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df['age_group'] = pd.cut(df.Age, range(0, 81, 10), right=False, labels=labels)
df[['Age', 'age_group']].drop_duplicates()[:10]


# In[25]:


#Visualize overall rating by users
df['Ratings'].value_counts().plot(kind='bar',alpha=0.7,figsize=(10,10))
plt.show()


# In[26]:


groupedby_movieName = df.groupby('MovieName')
groupedby_rating = df.groupby('Ratings')
groupedby_uid = df.groupby('UserID')
#groupedby_age = df.loc[most_50.index].groupby(['MovieName', 'age_group'])


# In[27]:


movies = df.groupby('MovieName').size().sort_values(ascending=True)[:1000]
print(movies)


# In[28]:


ToyStory_data = groupedby_movieName.get_group('Toy Story 2 (1999)')
ToyStory_data.shape


# In[29]:


#Find and visualize the user rating of the movie “Toy Story”
plt.figure(figsize=(10,10))
plt.scatter(ToyStory_data['MovieName'],ToyStory_data['Ratings'])
plt.title('Plot showing  the user rating of the movie “Toy Story”')
plt.show()


# In[31]:


#Find and visualize the viewership of the movie “Toy Story” by age group
ToyStory_data[['MovieName','age_group']]


# In[32]:


#Find and visualize the top 25 movies by viewership rating
top_25 = df[25:]
top_25['Ratings'].value_counts().plot(kind='barh',alpha=0.6,figsize=(7,7))
plt.show()


# In[33]:


#Visualize the rating data by user of user id = 2696
userid_2696 = groupedby_uid.get_group(2696)
userid_2696[['UserID','Ratings']]


# In[36]:


#First 500 extracted records
first_500 = df[500:]
first_500.dropna(inplace=True)


# In[37]:


#Use the following features:movie id,age,occupation
features = first_500[['MovieID','Age','Occupation']].values


# In[38]:


#Use rating as label
labels = first_500[['Ratings']].values


# In[48]:


# machine learning
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[40]:


#Create train and test data set
train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.33,random_state=4321)


# In[41]:


#Create a histogram for movie
df.Age.plot.hist(bins=25)
plt.title("Movie & Rating")
plt.ylabel('MovieID')
plt.xlabel('Ratings')


# In[42]:


#Create a histogram for age
df.Age.plot.hist(bins=25)
plt.title("Age & Rating")
plt.ylabel('Age')
plt.xlabel('Ratings')


# In[43]:


#Create a histogram for occupation
df.Age.plot.hist(bins=25)
plt.title("Occupation & Rating")
plt.ylabel('Occupation')
plt.xlabel('Ratings')


# In[52]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train, train_labels)
Y_pred = decision_tree.predict(test)
acc_decision_tree = round(decision_tree.score(train, train_labels) * 100, 2)
acc_decision_tree


# In[51]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train, train_labels)
Y_pred = random_forest.predict(test)
random_forest.score(train, train_labels)
acc_random_forest = round(random_forest.score(train, train_labels) * 100, 2)
acc_random_forest


# In[49]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(train, train_labels)
Y_pred = sgd.predict(test)
acc_sgd = round(sgd.score(train, train_labels) * 100, 2)
acc_sgd


# In[50]:


# Support Vector Machines

svc = SVC()
svc.fit(train, train_labels)
Y_pred = svc.predict(test)
acc_svc = round(svc.score(train, train_labels) * 100, 2)
acc_svc


# In[53]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines',  
              'Random Forest',   
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_svc,   
              acc_random_forest,  
              acc_sgd,  acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




