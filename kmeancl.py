import pandas as pd
import numpy as np
import matplotlib.pyplot  as pt

data=pd.read_csv("Mall_Customers.csv")
X=data.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss= []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10,max_iter=300,
                 random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

pt.plot(range(1,11),wcss)
pt.show()

kmeans=KMeans(n_clusters=5,init="k-means++",n_init=10,random_state=0)
y_pred=kmeans.fit_predict(X)

pt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c="blue",label="Careful")
pt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c="green",label="Standard")
pt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c="yellow",label="Target")
pt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c="red",label="Careless")
pt.scatter(X[y_pred==4,0],X[y_pred==4,1],s=100,c="brown",label="No")
pt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300
           ,c="black",label="Centroid")
pt.title("CLustered Data")
pt.xlabel("Income")
pt.ylabel("Score")
pt.legend()
pt.show()




