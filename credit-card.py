# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:16:44 2023

@author: stimp
"""

#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
credit_data = pd.read_csv("C:/Users/stimp/OneDrive/Desktop/Flame/OPSM322/OPSM322_Homework-1/CC_GENERAL.csv")

#finding and removing categorical variables
categorical_variables = credit_data.select_dtypes(include=['object'])
credit_data.drop(columns=categorical_variables, inplace=True)

#identify and replace missing values with averages
missing_values = credit_data.isnull().sum()
average_values = credit_data.mean()
for column in credit_data.columns:
    if missing_values[column] > 0:
        credit_data[column].fillna(average_values[column], inplace=True)

#scaling the data using MinMaxScaler
scaler = MinMaxScaler()
numeric_data = credit_data.select_dtypes(include=['number'])
scaler.fit(numeric_data)
scaled_data = scaler.transform(numeric_data)
scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)

#running K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(scaled_df)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

#calculating WCSS
wcss = kmeans.inertia_
print("WCSS:", wcss)

#elbow plot
wcss = []
for k in range(1, 26):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 26), wcss, marker='o', linestyle='--')
plt.title('Elbow Plot')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.xticks(np.arange(1, 26, step=1))
plt.grid(True)
plt.show()

#calculating silhouette score
silhouette_scores = []
k_values = range(2, 26)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(scaled_df)
    silhouette_avg = silhouette_score(scaled_df, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.grid(True)
plt.xticks(np.arange(2, 26, step=1))
plt.show()
#using the elbow plot and the silhouette scores we can conclude that the best value for K is 2

#running K-Means with the best value of K
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(scaled_df)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

#assigning cluster labels
scaled_df['cluster_id'] = cluster_labels

#scatterplot
X = scaled_df['PURCHASES']
Y = scaled_df['MINIMUM_PAYMENTS']
hue_variable = scaled_df['cluster_id']

sns.scatterplot(x=X, y=Y, hue=hue_variable)
plt.xlabel('Purchases')
plt.ylabel('Minimum Payments')
plt.title('Cluster ID as Hue')

plt.show()

#profiling the clusters using bar-plot
cluster_stats = scaled_df.groupby(cluster_labels).mean()
plt.figure(figsize=(8, 5))
cluster_stats.plot(kind='bar', colormap='viridis', rot=0)
plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.title('Bar-Plot')
leg = plt.legend(title='Feature', fontsize='xx-small', bbox_to_anchor=(0.5, -0.15), loc='upper center')

plt.show()

#profiling the clusters using scatter-plot
plt.figure(figsize=(10, 6))
for cluster in range(kmeans.n_clusters):
    cluster_data = scaled_df[cluster_labels == cluster]
    plt.scatter(cluster_data['CREDIT_LIMIT'], cluster_data['BALANCE'], label=f'Cluster {cluster + 1}')

plt.xlabel('Credit limit')
plt.ylabel('Balance')
plt.title('Cluster Profiling Using Scatterplot')
plt.legend()
plt.show()

#profiling the clusters box-plot
features = ['BALANCE','CREDIT_LIMIT','PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','PAYMENTS','TENURE']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, start=1):
    plt.subplot(3, 4, i)  # Adjust the subplot grid as needed
    sns.boxplot(x='cluster_id', y=feature, data=scaled_df)
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.title(f'Cluster Profiling Using Boxplot for {feature}')

plt.tight_layout()
plt.show()

