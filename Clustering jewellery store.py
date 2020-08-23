#!/usr/bin/env python
# coding: utf-8

# # [Shanshan, Lin]
# # [10065474]
# # [MMA]
# # [MMA 869]
# # [2020-08-08]

# # Perform a k-mean clustering analysis of the dataset. 

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
import tkinter as tk

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

mpl.use('TkAgg')


import pandas_profiling

import itertools

import scipy

from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer

from kmodes.kmodes import KModes

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# This will ensure that matplotlib figures don't get cut off when saving with savefig()
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# In[12]:


##Read in Data:


df = pd.read_csv("jewelry_customers.csv")


list(df)
df.shape
df.info()
df.describe().transpose()
df.head(n=20)
df.tail()


# In[13]:



###Profile the Data

pandas_profiling.ProfileReport(df, check_correlation=False)


# In[14]:



## Normalize the Data:

X = df.copy()
scaler = StandardScaler()
features = ['Age', 'Income','SpendingScore','Savings']
X[features] = scaler.fit_transform(X[features])

X.shape
X.info()
X.describe().transpose()
X.head(10)
X.tail()


# # Answer to Question [1], Part [b]

# In[15]:



###############K-Means################################################

k_means = KMeans(init='k-means++', n_clusters=5, n_init=10, random_state=101)
k_means.fit(X)

k_means.labels_

# Let's look at the centers
k_means.cluster_centers_


# In[16]:


##Internal Validation Metrics

###when k=5:
# WCSS == Inertia:
k_means.inertia_

#Silhouette Score:
silhouette_score(X, k_means.labels_)

###when k=3:
k_means3 = KMeans(init='k-means++', n_clusters=3, n_init=10, random_state=101)
k_means3.fit(X)

k_means3.inertia_
silhouette_score(X, k_means3.labels_)

###when k=7:
k_means7 = KMeans(init='k-means++', n_clusters=7, n_init=10, random_state=101)
k_means7.fit(X)
k_means7.inertia_
silhouette_score(X, k_means7.labels_)

###when k=10:
k_means10 = KMeans(init='k-means++', n_clusters=10, n_init=10, random_state=101)
k_means10.fit(X)
k_means10.inertia_
silhouette_score(X, k_means10.labels_)



# In[17]:



####Elbow Method 
###### according to the elbow graphes below, k=5 is indeed the best value which result in lowest WSCC and highest silhoettes score

inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, max_iter=1000, random_state=42).fit(X)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(X, kmeans.labels_, metric='euclidean')
    

plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");
plt.savefig('869Individual_AssignmentQ1_graphs/jelwery-kmeans-elbow-interia.png');


plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");
plt.savefig('869Individual_AssignmentQ1_graphs/jelwery-kmeans-elbow-silhouette.png');


# In[18]:


###sklearn.metrics.davies_bouldin_score(X, k_means.labels_)

visualizer = SilhouetteVisualizer(k_means)
visualizer.fit(X)
visualizer.poof()
fig = visualizer.ax.get_figure()
fig.savefig('869Individual_AssignmentQ1_graphs/jelwery-kmeans-5-silhouette.png', transparent=False);


# # Answer to Question [1], Part [c]

# In[19]:



###Intepretting the Clusters


###Means
k_means.cluster_centers_

###print summary statistics for each cluster
for label in set(k_means.labels_):
    print('\nCluster {}:'.format(label))
    X_tmp = X[k_means.labels_==label].copy()
    X_tmp.loc['mean'] = X_tmp.mean()
    X_tmp.describe() 



# In[20]:


###Find Examplars

from scipy.spatial import distance

for i, label in enumerate(set(k_means.labels_)):
    X_tmp = X[k_means.labels_ == label].copy()

    exemplar_idx = distance.cdist([k_means.cluster_centers_[i]], X_tmp).argmin()
    exemplar = pd.DataFrame(X_tmp.iloc[exemplar_idx])

    print('\nCluster {}:'.format(label))
    exemplar


# In[21]:


###Relative Importance Plots


dat = X.copy()
dat['Cluster'] = k_means.labels_

# Calculate average values for each cluster
cluster_avg = dat.groupby(['Cluster']).mean()

# Calculate average values for the total population
population_avg = dat.drop(['Cluster'], axis=1).mean()

# Calculate relative importance of cluster's attribute value compared to population
relative_imp = cluster_avg - population_avg

# Initialize a plot with a figure size of 8 by 4 inches 
plt.figure(figsize=(8, 4));
plt.title('Relative importance of features');
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn');

###to fix the ylim cutoff issue appeared:
b, t = plt.ylim() # discover the values for bottom and top
b += 0.4 # Add 0.4 to the bottom
t -= 0.4 # Subtract 0.4 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()

plt.savefig('869Individual_AssignmentQ1_graphs/jelwery-kmeans-5-importance.png', transparent=False);

