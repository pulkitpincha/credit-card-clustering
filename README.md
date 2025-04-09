# credit-card-clustering

Dataset: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

Credit card user clustering using K-Means and profiling them

## Data Wrangling

- Removing Categorical Variables
- Identify and replace missing values with averages
- Scaling the data using MinMaxScaler

## Scatter-plot Analysis

The scatter plot effectively displays how the data points are distributed among the clusters. 
It's evident that the majority of data points are concentrated around 0.6 on both the x and y axes. 
Consequently, there is significant overlap among the clusters, and they are not evenly spaced 
apart. It appears that both variables have an approximately equal impact on the formation of 
these clusters.

## Cluster Profiles

The 7 clusters are as follows:
- Cluster 1: The most significant variables are Balance Frequency, Cash Advance, and 
Tenure.
- Cluster 2: The primary influential variables are Balance Frequency and Tenure.
- Cluster 3: The most influential variables are Balance Frequency, Cash Advance, and 
Tenure.
- Cluster 4: The most influential variable is Tenure.
- Cluster 5: The most influential variable is Balance Frequency.
- Cluster 6: The most significant variables are Balance Frequency, Cash Advance, and 
Tenure.
- Cluster 7: The most significant variables are Balance Frequency, Cash Advance, and 
Tenure.

# Analysis

- The most important variables include Balance Frequency, Cash Advance, and Length of 
Membership.
- The clusters appear to be indistinct and not feasible, evident from both the scatterplot and 
profiling. Significant overlap among the clusters is observed, primarily in relation to three 
variables for each of them.
