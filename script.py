import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("Mall_Customers.csv")

x = df.iloc[:,[3,4]].values

#get number of clusters
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

#elbow graph
sb.set()
plt.plot(range(1,11),wcss)
plt.title("Elbow graph")
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.show()

#use graph to determine value of n

#model
km = KMeans(n_clusters=5,init="k-means++",random_state=42)

y = km.fit_predict(x)

#visualize 
#plot n cluster points manually
plt.figure(figsize=(10,10))
plt.scatter(x[y==0,0],x[y==0,1],s=50,c='green',label='c1')
plt.scatter(x[y==1,0],x[y==1,1],s=50,c='red',label='c2')
plt.scatter(x[y==2,0],x[y==2,1],s=50,c='yellow',label='c3')
plt.scatter(x[y==3,0],x[y==3,1],s=50,c='blue',label='c4')
plt.scatter(x[y==4,0],x[y==4,1],s=50,c='orange',label='c5')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=100,c='cyan',label='centroid')
plt.title("Customer groups")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.show()