import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('data.csv')

# Select relevant features for clustering (example: CreditScore and Age)
features = data[['CreditScore', 'Age']]

# Specify the number of clusters (example: 3 clusters)
num_clusters = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)

# Visualize the clusters
plt.scatter(data['CreditScore'], data['Age'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Credit Score')
plt.ylabel('Age')
plt.show()