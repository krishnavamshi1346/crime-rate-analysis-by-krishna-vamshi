#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import pickle


# In[2]:


os.getcwd()


# In[3]:


data=pd.read_csv("t10-crime.csv")
data


# In[4]:


data.columns


# DATA REFINEMENT

# In[5]:


data.isnull().sum()


# In[6]:


mode=data['UCR_PART'].mode()
print(mode)


# In[7]:


data ['UCR_PART'].fillna('Part Three', inplace=True)


# In[8]:


mode=data['DISTRICT'].mode()
print(mode)


# In[9]:


#cols = [0]
#data[cols] = data[cols].fillna(data.mode().iloc[0])
data ['DISTRICT'].fillna('B2', inplace=True)


# In[10]:


data.drop(['STREET'],axis=1,inplace=True)


# In[11]:


data.drop(['Lat'],axis=1,inplace=True)


# In[12]:


data.drop(['Long'],axis=1,inplace=True)


# In[13]:


data.drop(['Location'],axis=1,inplace=True)


# In[14]:


data.isnull().sum()


# EXPLORATARY DATA ANALYSIS

# In[15]:


data.describe()


# In[16]:


data.corr()


# In[17]:


data.cov()


# In[18]:


data.head()


# In[19]:


data.tail()


# In[20]:


data.drop(['INCIDENT_NUMBER'],axis=1,inplace=True)


# data.drop(['﻿INCIDENT_NUMBER'],axis=1,inplace=True)

# In[21]:


data[data["UCR_PART"].str.contains("Part Three")]


# In[22]:


hour=data['HOUR']
hour


# In[23]:


ucr=data['UCR_PART']
ucr


# In[24]:


data.drop(['OFFENSE_CODE'],axis=1,inplace=True)


# In[25]:


data.drop(['OFFENSE_CODE_GROUP'],axis=1,inplace=True)


# In[26]:


data.drop(['OFFENSE_DESCRIPTION'],axis=1,inplace=True)


# In[27]:


data.drop(['REPORTING_AREA'],axis=1,inplace=True)


# In[28]:


data.drop(['OCCURRED_ON_DATE'],axis=1,inplace=True)


# In[29]:


data.drop(['SHOOTING'],axis=1,inplace=True)


# In[30]:


data


# In[31]:



data['DAY_OF_WEEK'].replace(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],[1,2,3,4,5,6,7],inplace=True)


# In[33]:



data['UCR_PART'].replace(['Part One',"Part Two","Part Three","Other"],[1,2,3,4],inplace=True)


# In[32]:



data['DISTRICT'].replace(['B2','C11','D4','A1','B3','C6','D14','E13','E18','A7','E5','A15'],[1,2,3,4,5,6,7,8,9,10,11,12],inplace=True)


# In[34]:


data


# In[35]:


import  matplotlib.pyplot as plt 
#get_ipython().run_line_magic('matplotlib', 'inline')
data['UCR_PART'].hist(bins=10)


# In[36]:



data['MONTH'].hist(bins=12)


# In[37]:



data['HOUR'].hist(bins=24)


# In[38]:


data.boxplot(column='UCR_PART')


# In[39]:


data["UCR_PART"].value_counts()


# In[40]:


data.boxplot(column="UCR_PART",by='HOUR')


# In[41]:


data.boxplot(column="UCR_PART",by='DAY_OF_WEEK')


# In[42]:


data.boxplot(column="UCR_PART",by='DISTRICT')


# In[43]:


data.boxplot(column="UCR_PART",by='YEAR')


# In[44]:


data.boxplot(column="UCR_PART",by='MONTH')


# In[45]:


pd.crosstab(data['UCR_PART'],data['DISTRICT'],margins=True)


# In[46]:


pd.crosstab(data['UCR_PART'],data['YEAR'],margins=True)


# In[49]:


import seaborn as sns
sns.pairplot(data)


# In[50]:


# Plot the correlation usinf heatmap
corr = data.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)


# REGRESSION ANALYSIS

# LINEAR REGRESSION

# In[47]:


x=data['DISTRICT']


# In[48]:


y=data['UCR_PART']


# In[49]:


from sklearn.model_selection import train_test_split  #train_test_split splits arrays or matrices into random train and test subsets. 
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4,random_state =1) 


# In[50]:


x_train


# In[51]:


x_test


# In[52]:


y_test


# In[53]:


x_train = x_train.values.reshape(-1,1)
x_train


# In[54]:


from sklearn import linear_model as lm
model=lm.LinearRegression()
results=model.fit(x_train,y_train) 


# In[55]:


accuracy = model.score(x_train, y_train)
print('Accuracy of the model:', accuracy)


# In[56]:


#Print coefficients
print('intercept:', model.intercept_)
print('slope:', model.coef_)


# In[57]:


model


# In[58]:


#Predictions from the model
x_test = x_test.values.reshape((-1,1))#if x_test.reshape((-1,-1)) doesn't work use this

predictions = model.predict(x_test)
print('predicted ucr_part:',predictions, sep = '\n')


# In[59]:


#Visualize the predictions
plt.scatter(y_test, predictions)


# In[60]:


#Evaluating the model 
from sklearn.metrics import mean_squared_error, r2_score
x_train = x_train.reshape(-1,1)
y_train_prediction = model.predict(x_train)

x_test = x_test.reshape(-1,1)
y_test_prediction = model.predict(x_test)


# In[61]:


# printing values
print('Slope:' ,model.coef_)
print('Intercept:', model.intercept_)
print("\n")

# model evaluation for training set
#import numpy as np
rmse_training = (np.sqrt(mean_squared_error(y_train, y_train_prediction)))
r2_training = r2_score(y_train, y_train_prediction)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse_training))#rmse root mean sqaured error 
print('R2 score is {}'.format(r2_training))
print("\n")

# model evaluation for testing set
rmse_testing = (np.sqrt(mean_squared_error(y_test, y_test_prediction)))
r2_testing = r2_score(y_test, y_test_prediction)

print("The model performance for testing set")
print("--------------------------------------")
print('Root mean squared error: ', rmse_testing)
print('R2 score: ', r2_testing)


# MULTIPLE LINEAR REGERSSION

# In[62]:


X=data[['DISTRICT','YEAR','DAY_OF_WEEK','HOUR']]


# In[63]:


Y=data[['UCR_PART']]


# In[64]:


X


# In[65]:


Y


# In[66]:


data["DISTRICT"]=data["DISTRICT"].astype(int)


# In[67]:


data["UCR_PART"]=data["UCR_PART"].astype(int)


# In[68]:


data["YEAR"]=data["YEAR"].astype(int)


# In[69]:


data["DAY_OF_WEEK"]=data["DAY_OF_WEEK"].astype(int)


# In[70]:


data["HOUR"]=data["HOUR"].astype(int)


# In[71]:


import statsmodels.api as sm
model1=sm.OLS(Y,X).fit()
predictions = model1.predict(X)
model1.summary()


# In[72]:


# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2, random_state=1)


# In[73]:


X_train


# In[74]:


Y_train


# In[75]:


# define the data/predictors as the pre-set feature names  
features = X_train.iloc[:,:].values


# In[76]:


labels = Y_train.iloc[:].values


# In[77]:


# Instantiate Multiple linear regrssion model
from sklearn import linear_model as lm
model=lm.LinearRegression()
results=model.fit(X,Y) 


# In[78]:


predictions = model.predict(X)


# In[79]:


#Check model accuracy
accuracy=model.score(X,Y)
print('Accuracy of the model:', accuracy)


# KNN CLASSIFIER

# In[80]:


from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)


# In[81]:


#Predict on test data
y_pred=clf.predict(X_test)
y_pred


# In[82]:


print("Actual y values : ")
print(Y_test.values)


# In[83]:


#Accuracy score on Test and Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(Y_test,y_pred) * 100))
#print("Recall score : %f" %(recall_score(Y_test, y_pred) * 100))


# LOGISTIC REGRESSION

# In[84]:


#Build a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)


# In[85]:


#Build a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)


# In[86]:


#Accuracy score on Test and Train
from sklearn.metrics import accuracy_score, recall_score

print("\nAccuracy score: %f" %(accuracy_score(Y_test,y_pred) * 100))
#print("Recall score : %f" %(recall_score(Y_test, y_pred) * 100))


# RANDOM FOREST

# In[87]:


from sklearn.ensemble import RandomForestClassifier
RFmodel=RandomForestClassifier(n_estimators = 10,
                            random_state=0).fit(X_train,Y_train)
RFmodel.predict(X_test)
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(RFmodel.score(X_train,Y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'

      .format(RFmodel.score(X_test,Y_test)))


# # RESULTS

# The accuracy of Multiple Linear Regression on test set is:4

# The accuracy of KNN classifier on test set is:40 

# The accuracy of Logistic regression on test set is:40.15

# The accuracy of Random Forest classifier on test set is:51

# So, We conclude that the accuracy among KNN,Logistic Regression,Random Forest Classifier the accuracy is greater for Random Forest with an accuracy of  51 percent

# In[89]:


pickle.dump(RFmodel,open('model.pkl','wb'))


# In[ ]:




