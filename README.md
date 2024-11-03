# Clustering and Anomaly Detection in Wine Quality

## Overview

This project focuses on clustering and anomaly detection using the white wine quality dataset from the Portuguese "Vinho Verde" wine. The goal is to extract knowledge from the data by applying various machine learning techniques. This analysis will explore the physiochemical properties of the wine and its quality rating on a scale of 1-10.

## Table of Contents

- [Data Description](#data-description)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Data Analysis](#data-analysis)
  - [Univariate Analysis](#univariate-analysis)
  - [Bivariate Analysis](#bivariate-analysis)
  - [Outlier Detection](#outlier-detection)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Clustering](#clustering)
  - [K-Means Clustering](#k-means-clustering)
  - [Density-Based Clustering (DBSCAN)](#density-based-clustering-dbscan)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Data Description

The dataset used in this project consists of 4,898 instances and 11 features, representing the physiochemical properties of white wine. The last column indicates the quality of the wine, rated between 1 and 10. The analysis focuses solely on the white wine dataset due to its larger sample size, which is expected to yield better model performance.

## Libraries Used

This project utilizes the following Python libraries for data analysis and visualization:

- `pandas`: For data manipulation and analysis
- `matplotlib`: For creating visualizations
- `seaborn`: For enhanced visualizations
- `scikit-learn`: For implementing machine learning algorithms, including PCA and clustering methods

## Data Preprocessing

- The "Quality" label was removed to focus on unsupervised learning.
- Duplicate entries were eliminated to ensure data integrity.
- Data types were examined using `head()`, `tail()`, and `info()` methods.
- Univariate statistics were computed using the `describe()` method in Pandas to summarize key characteristics of the dataset.

## Data Analysis

### Univariate Analysis

Univariate analysis was performed to understand the distribution of individual variables. Histograms and kernel density plots were utilized to visualize the distribution, identifying skewness and kurtosis.

### Bivariate Analysis

Bivariate analysis explored the relationships between two variables, employing scatter plots and correlation matrices to identify potential connections.

### Outlier Detection

Outliers were visualized using boxplots, and methods like the z-score and IQR were applied to identify and assess the impact of outliers on the dataset.

## Dimensionality Reduction

Dimensionality reduction techniques were employed to reduce the feature space. Principal Component Analysis (PCA) was applied to transform the data while retaining essential information, allowing for effective visualization in 2D and 3D.

## Clustering

### K-Means Clustering

K-Means clustering was performed to group similar data points. The optimal number of clusters (k) was determined using metrics such as inertia, Davies-Bouldin score, Calinski-Harabasz score, and Silhouette score.

### Density-Based Clustering (DBSCAN)

DBSCAN was applied to detect clusters of varying shapes and identify noise points within the data, revealing insights into the underlying data distribution.

## Results

The analysis produced various insights into the wine quality dataset, including clustering patterns and potential anomalies. The findings indicated that the clustering of wine quality may not align perfectly with the true quality scores due to the limited range of unique values.

## Conclusion

This project demonstrated the effectiveness of clustering and anomaly detection techniques in understanding the wine quality dataset. Further analysis could explore additional machine learning methodologies or different datasets to refine the results.

## Acknowledgments

- UCI Machine Learning Repository for hosting the wine quality dataset.
- Documentation and resources for libraries such as Pandas, Matplotlib, Seaborn, and Scikit-Learn.
