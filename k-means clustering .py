#!/usr/bin/env python
# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
%matplotlib inline
df=pd.read_csv("Live.csv")
# In[41]:


df.head()


# <h4>Check shape of the dataset

# In[42]:


df.shape


# We can see that there are 7050 instances and 16 attributes in the dataset. In the dataset description, it is given that there are 7051 instances and 12 attributes in the dataset.
# 
# So, we can infer that the first instance is the row header and there are 4 extra attributes in the dataset. Next, we should take a look at the dataset to gain more insight about 

# <h4>View summary of dataset

# In[43]:


df.info()


# <h4>Check for missing values in dataset

# In[44]:


df.isnull().sum()


# We can see that there are 4 redundant columns in the dataset. We should drop them before proceeding further.

# <h4>Drop redundant columns

# In[45]:


df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)


# <h4>Again view summary of dataset

# In[46]:


df.info()


# Now, we can see that redundant columns have been removed from the dataset.
# 
# We can see that, there are 3 character variables (data type = object) and remaining 9 numerical variables (data type = int64).

# <h4>View the statistical summary of numerical variables

# In[47]:


df.describe()


# There are 3 categorical variables in the dataset. I will explore them one by one.

# <h4>Explore status_id variable

# In[48]:


df['status_id'].unique()


# In[49]:


len(df['status_id'].unique())


# We can see that there are 6997 unique labels in the status_id variable. The total number of instances in the dataset is 7050. So, it is approximately a unique identifier for each of the instances. Thus this is not a variable that we can use. Hence, I will drop it.

# In[50]:


df['status_published'].unique()


# In[51]:


len(df['status_published'].unique())


# Again, we can see that there are 6913 unique labels in the status_published variable. The total number of instances in the dataset is 7050. So, it is also a approximately a unique identifier for each of the instances. Thus this is not a variable that we can use. Hence, I will drop it also.

# <h4>Explore status_type variable

# In[52]:


df['status_type'].unique()


# In[53]:


len(df['status_type'].unique())


# We can see that there are 4 categories of labels in the status_type variabl

# <h4>Drop status_id and status_published variable from the dataset

# In[54]:


df.drop(['status_id', 'status_published'], axis=1, inplace=True)


# <h4>View the summary of dataset again

# In[55]:


df.info()


# # Declare feature vector and target variable 

# In[56]:


X = df
y = df['status_type']


# # Convert categorical variable into integers 

# In[57]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['status_type'] = le.fit_transform(X['status_type'])
y = le.transform(y)


# # Feature Scaling

# In[58]:


cols = X.columns


# In[59]:


from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X = ms.fit_transform(X)


# In[60]:


X = pd.DataFrame(X, columns=[cols])


# In[61]:


X.head()


# # K-Means model with two clusters

# In[62]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0) 
kmeans.fit(X)


# <h4>K-Means model parameters stud

# In[63]:


kmeans.cluster_centers_


# The KMeans algorithm clusters data by trying to separate samples in n groups of equal variances, minimizing a criterion known as inertia, or within-cluster sum-of-squares Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are.
# The k-means algorithm divides a set of N samples X into K disjoint clusters C, each described by the mean j of the samples in the cluster. The means are commonly called the cluster centroids.
# The K-means algorithm aims to choose centroids that minimize the inertia, or within-cluster sum of squared criterion.

# In[64]:


kmeans.inertia_


# # Check quality of weak classification by the model

# In[65]:


labels = kmeans.labels_
# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))


# In[66]:


print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# We have achieved a weak classification accuracy of 1% by our unsupervised model.

# <h4> Use elbow method to find optimal number of clusters

# In[67]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# By the above plot, we can see that there is a kink at k=2.
# 
# Hence k=2 can be considered a good number of the cluster to cluster this data.
# 
# But, we have seen that I have achieved a weak classification accuracy of 1% with k=2.
# 
# I will write the required code with k=2 again for convinience.

# In[68]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2,random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# So, our weak unsupervised classification model achieved a very weak classification accuracy of 1%.
# 
# I will check the model accuracy with different number of clusters.

# # K-Means model with different clusters

# In[69]:


kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[70]:


kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# We have achieved a relatively high accuracy of 62% with k=4.

# In this project, I have implemented the most popular unsupervised clustering technique called K-Means Clustering.
# 
# I have applied the elbow method and find that k=2 (k is number of clusters) can be considered a good number of cluster to cluster this data.
# 
# I have find that the model has very high inertia of 237.7572. So, this is not a good model fit to the data.
# 
# I have achieved a weak classification accuracy of 1% with k=2 by our unsupervised model.
# 
# So, I have changed the value of k and find relatively higher classification accuracy of 62% with k=4.
# 
# Hence, we can conclude that k=4 being the optimal number of clusters.

# In[ ]:




