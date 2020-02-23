# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:43:10 2020

@author: Vanshita
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

# Load the data
raw_data = pd.read_csv('D://sih//andaman.csv')
# Check the data
raw_data
data = raw_data.copy()

plt.scatter(data['Area'], data['Production'])
plt.xlim(-10,200)
plt.ylim(-10, 110)
plt.show()

x = data.iloc[:]
x

kmeans = KMeans(6)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
identified_clusters
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters
plt.scatter(data['Area'], data['Production'],c=data_with_clusters['Cluster'], cmap = 'rainbow')
plt.xlim(-10,200)
plt.ylim(-10, 110)
plt.show()