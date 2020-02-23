#!/usr/bin/env python
# coding: utf-8

# SIMPLE LINEAR REGRESSION
# 

# Import the relevant libraries

# In[121]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import io
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the data

# In[122]:


raw_data = pd.read_csv('D://sih//andaman.csv')


# In[123]:


crops_name = ['Arecanut', 'Other Kharif pulses', 'Rice', 'Banana', 'Cashewnut',
        'Coconut ', 'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca',
       'Black pepper', 'Dry chillies', 'other oilseeds', 'Turmeric',
       'Maize', 'Moong(Green Gram)', 'Urad', 'Arhar/Tur', 'Groundnut',
       'Sunflower']


# In[124]:


labels = list(range(20))


# In[125]:


dict1 = {}


# In[126]:


for i,j in zip(crops_name, labels):
    dict1[i] = j    


# In[127]:


raw_data['label_crop'] = raw_data['Crop'].map(dict1)


# In[128]:


season_name = raw_data['Season'].unique()


# In[129]:


labels = list(range(4))


# In[130]:


for i, j in zip(season_name, labels):
    dict1[i] = j


# In[131]:


dict1


# In[132]:


raw_data['label_season'] = raw_data['Season'].map(dict1)


# In[133]:


raw_data.head()


# In[134]:


raw_data.info()


# 
# CREATING FIRST REGRESSION

# Define the dependent and the independent variable

# In[135]:


y = raw_data['Season']
x1 = raw_data['District_Name']


# Explore the data

# In[136]:


plt.scatter(x1,y)
plt.xlabel('District_Name', fontsize=20)
plt.ylabel('Season', fontsize=20)
plt.show()


# Regression Itself

# In[137]:


data = raw_data.copy()


# In[138]:


plt.scatter(data['Season'], data['Crop'])
plt.xlim(-2,4)
plt.ylim(-1, 20)
plt.show()


# In[139]:


x2 = data['label_season'] 
y2 = data['label_crop']


# In[140]:


x_matrix = x2.values.reshape(-1,1) #as x is a vector of 1D and sklearn supports 2D we are transforming it to a matrix
x_matrix.shape


# In[141]:


reg = LinearRegression()


# In[142]:


reg.fit(x_matrix,y2) 


# In[143]:


reg.score(x_matrix,y2) 


# In[144]:


reg.coef_


# In[145]:


reg.intercept_


# In[146]:


new_data = pd.DataFrame(data,columns = ['label_season'])
new_data


# In[147]:


reg.predict(new_data)


# In[148]:


prediction = reg.predict(new_data)


# In[149]:


new_data


# In[150]:


len(prediction)


# In[151]:


np.round(new_data)


# In[152]:


crops_name[int(np.round(reg.predict([[0]]))[0])]


# In[153]:


x3 = data['Crop']
y3 = data['Production']


# In[154]:


plt.scatter(x3,y3)
plt.xlim(-1,20)
plt.rc('xtick', labelsize=10)  
plt.xticks(rotation=90)
plt.ylabel('Production', fontsize=20)
plt.xlabel('Crop', fontsize=20)
plt.show()


# In[155]:


x4 = data['label_crop']


# In[156]:


x_matrixs = x4.values.reshape(-1,1) 
x_matrixs.shape


# In[157]:


reg1 = LinearRegression()


# In[158]:


reg1.fit(x_matrixs,y3) 


# In[165]:


reg1.score(x_matrixs,y3) 


# In[166]:


reg1.coef_


# In[167]:


reg1.intercept_


# In[168]:


neww_data = pd.DataFrame(data, columns = ['label_crop'])
neww_data


# In[169]:


reg1.predict(neww_data)


# In[170]:


predictionn = reg1.predict(neww_data)
neww_data


# In[173]:


reg1.predict([[2]])


# In[ ]:




