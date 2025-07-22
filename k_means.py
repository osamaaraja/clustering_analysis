# ==================================================
#     K-MEANS Clustering using 'euclidean' metric
# ==================================================

# Creating a copy of final pre_processed dataframe to use it for K-Means clustering.
k_means_data=preprocessed_data.copy()
print(k_means_data.shape)
k_means_data.head()

# Importing necessary libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Extracting the features (identifiers are excluded)
feature_cols = [col for col in k_means_data.columns if col not in ['hadm_id', 'subject_id']]
X = k_means_data[feature_cols].values

# Initializing lists to store metrics
silhouette_scores = []
calinski_scores = []
davies_scores = []
k_values = list(range(2, 16))

# Store all metrics for each k
metrics_results = []

for k in k_values:
    print(f"Clustering with k={k}...")

    # Training K-means model and predicting labels accordingly
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Computing metrics
    silhouette_val = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_val)

    calinski_val = calinski_harabasz_score(X, cluster_labels)
    calinski_scores.append(calinski_val)

    davies_val = davies_bouldin_score(X, cluster_labels)
    davies_scores.append(davies_val)

    # Storing results
    metrics_results.append({
        'k': k,
        'silhouette': silhouette_val,
        'calinski_harabasz': calinski_val,
        'davies_bouldin': davies_val
    })

# Converting results into a DataFrame
metrics_df = pd.DataFrame(metrics_results)

# Plotting scores
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(k_values, metrics_df['silhouette'], marker='o', color='red')
plt.grid()
plt.title('Silhouette Score')
plt.xlabel('Number of clusters (K)')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(k_values, metrics_df['calinski_harabasz'], marker='o', color='blue')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of clusters (K)')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(k_values, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of clusters (K)')

plt.tight_layout()
plt.show()

best_silhouette_n = list(k_values)[np.nanargmax(silhouette_scores)]
best_calinski_n = list(k_values)[np.nanargmax(calinski_scores)]
best_davies_n = list(k_values)[np.nanargmax(davies_scores)]

print(f'Optimal no. of clusters:{best_silhouette_n} (Silhouette Score: {np.nanmax(silhouette_scores):.3f}), Optimal no. of clusters:{best_calinski_n} (Calinski Score: {np.nanmax(calinski_scores):.3f}), Optimal no. of clusters:{best_davies_n} (Davies Score: {np.nanmax(davies_scores):.3f})')

metrics_df

# Identifying Top 3 Clusters
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold, if any
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ the pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette   score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['k'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest (no threshold)
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['k'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest (no threshold)
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['k'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop cluster numbers per metric:")
print(top_clusters)

# Fisher's Test on Selected Clusters

# Function to run Fisher's test
def run_fisher_test(X, k, outcome_data):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Merging cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': k_means_data['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

# Running Fisher's test for each top cluster configuration
print("\nRunning Fisher's tests on top-3 clusters...")
all_results = {}

for metric, clusters in top_clusters.items():
    if not clusters:
         print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
         continue
    for k in clusters:
        print(f"\nTesting k={k} (from {metric} metric)")
        results = run_fisher_test(X, k, cohort_data)
        all_results[f'{metric}_k={k}'] = results
        print(f"Results for k={k} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ============================================================================
#   K-Medoids Clustering using Manhattan Distance matrix (without Imputation)
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Calculating time to compute custom (Manhattan) distance matrix
start = time.time()

'''
Using Feature Matrix "feature_matrix" that we created above before applying Normalization/Imputation i.e

feature_cols = [col for col in pivot.columns if col not in ['hadm_id', 'subject_id']]
feature_matrix = pivot[feature_cols].values

'''
print("Computing Manhattan distance matrix...")

# Computing Manhattan distance using only overlapping non-missing features for each pair
def custom_manhattan_distance(X):

    n = X.shape[0]
    distance_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i+1, n):
            # Find overlapping non-missing features
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])

            if mask.sum() > 0:
                # Manhattan distance: sum of absolute differences
                dist = np.sum(np.abs(X[i, mask] - X[j, mask]))
                distance_matrix[i, j] = distance_matrix[j, i] = dist

    return distance_matrix

# Usage
print("Computing Manhattan distance matrix (overlapping labs only)...")
manhattan_dist = custom_manhattan_distance(feature_matrix)

# Preparing distance matrix for clustering
max_dist = np.nanmax(manhattan_dist)
manhattan_dist_filled = np.where(np.isnan(manhattan_dist), max_dist + 1, manhattan_dist)
np.fill_diagonal(manhattan_dist_filled, 0)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Manhattan distance matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)

for n_clusters in range_n_clusters:
    print(f"K-Medoids with {n_clusters} clusters...")
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(manhattan_dist_filled)

    # Computing metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (supports precomputed)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(manhattan_dist_filled, labels, metric='precomputed')

    # Since Calinski-Harabasz and Davies-Bouldin require feature space so We'll use the imputed/normalized version from preprocessed data
    try:
        # Using (imputed+normalized) feature coloumns
        X_preprocessed = preprocessed_data[feature_cols].values
        if len(np.unique(labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_preprocessed, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_preprocessed, labels)
    except Exception as e:
        print(f"  Error computing metrics: {str(e)}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Plotting Scores
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o', color='red')
plt.grid()
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='blue')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of clusters')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of clusters')

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette   score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTo-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': pivot['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
         print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
         continue
    for n in clusters:
        print(f"\nTesting n_clusters={n} (from {metric} metric)")
        kmedoids = KMedoids(n_clusters=n, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(manhattan_dist_filled)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for n={n} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ============================================================================
#   K-Medoids Clustering using Cosine Distance matrix (without Imputation)
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Calculating time to compute custom (Cosine) distance matrix
start = time.time()

'''
Using Feature Matrix "feature_matrix" that we created above before applying Normalization/Imputation i.e

feature_cols = [col for col in pivot.columns if col not in ['hadm_id', 'subject_id']]
feature_matrix = pivot[feature_cols].values

'''
print("Computing Cosine distance matrix...")

# Computing Cosine distance using only overlapping non-missing features for each pair
def custom_cosine_distance(X):

    n = X.shape[0]
    distance_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i+1, n):
            # Find overlapping non-missing features
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])

            if mask.sum() > 0:
                # Extract vectors for overlapping features
                vec_i = X[i, mask]
                vec_j = X[j, mask]

                # Compute cosine distance
                distance_matrix[i, j] = distance_matrix[j, i] = cosine_distances(
                    [vec_i], [vec_j]
                )[0][0]

    return distance_matrix

cosine_dist = custom_cosine_distance(feature_matrix)

# Preparing distance matrix for clustering
max_dist = np.nanmax(cosine_dist)
cosine_dist_filled = np.where(np.isnan(cosine_dist), max_dist + 1, cosine_dist)
np.fill_diagonal(cosine_dist_filled, 0)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Cosine distance matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)

for n_clusters in range_n_clusters:
    print(f"K-Medoids with {n_clusters} clusters...")
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(cosine_dist_filled)

    # Computing metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (supports precomputed)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(cosine_dist_filled, labels, metric='precomputed')

    # Since Calinski-Harabasz and Davies-Bouldin require feature space so We'll use the imputed/normalized version from preprocessed data
    try:
        # Using (imputed+normalized) feature coloumns
        X_preprocessed = preprocessed_data[feature_cols].values
        if len(np.unique(labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_preprocessed, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_preprocessed, labels)
    except Exception as e:
        print(f"  Error computing metrics: {str(e)}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Plotting Scores
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o', color='red')
plt.grid()
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='blue')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of clusters')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of clusters')

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette   score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTo-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': pivot['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
         print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
         continue
    for n in clusters:
        print(f"\nTesting n_clusters={n} (from {metric} metric)")
        kmedoids = KMedoids(n_clusters=n, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(cosine_dist_filled)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for n={n} ({metric}):")
        display(results)







#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ============================================================================
#   K-Medoids Clustering using Euclidean Distance matrix (without Imputation)
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Calculating time to compute custom (Euclidean) distance matrix
start = time.time()

'''
Using Feature Matrix "feature_matrix" that we created above before applying Normalization/Imputation i.e

feature_cols = [col for col in pivot.columns if col not in ['hadm_id', 'subject_id']]
feature_matrix = pivot[feature_cols].values

'''
print("Computing Euclidean distance matrix...")

# Computing Euclidean distance using only overlapping non-missing features for each pair
def custom_euclidean_distance(X):
    n = X.shape[0]
    distance_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i+1, n):
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])
            if mask.sum() > 0:
                dist = np.linalg.norm(X[i, mask] - X[j, mask])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix

euclidean_dist = custom_euclidean_distance(feature_matrix)

# Preparing distance matrix for clustering
max_dist = np.nanmax(euclidean_dist)
euclidean_dist_filled = np.where(np.isnan(euclidean_dist), max_dist + 1, euclidean_dist)
np.fill_diagonal(euclidean_dist_filled, 0)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Euclidean distance matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)

for n_clusters in range_n_clusters:
    print(f"K-Medoids with {n_clusters} clusters...")
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(euclidean_dist_filled)

    # Computing metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (supports precomputed)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(euclidean_dist_filled, labels, metric='precomputed')

    # Since Calinski-Harabasz and Davies-Bouldin require feature space so We'll use the imputed/normalized version from preprocessed data
    try:
        # Using (imputed+normalized) feature coloumns
        X_preprocessed = preprocessed_data[feature_cols].values
        if len(np.unique(labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_preprocessed, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_preprocessed, labels)
    except Exception as e:
        print(f"  Error computing metrics: {str(e)}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Plotting Scores
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o', color='red')
plt.grid()
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='blue')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of clusters')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of clusters')

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette   score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTo-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': pivot['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
         print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
         continue
    for n in clusters:
        print(f"\nTesting n_clusters={n} (from {metric} metric)")
        kmedoids = KMedoids(n_clusters=n, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(euclidean_dist_filled)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for n={n} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# =====================================================
#    K-Medoids Clustering using DTW Distance Matrix
# =====================================================

# Creating a copy of final pre_processed dataframe to use it for K-Medoids clustering.
k_mediods_data=preprocessed_data.copy()
print(k_mediods_data.shape)
k_mediods_data.head()

# Importing necesssary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import time

# Calculating time to compute DTW matrix
start = time.time()

# Loading preprocessed data (features)
feature_cols = [col for col in k_mediods_data.columns if col not in ['hadm_id', 'subject_id']]

# Preparing Data for DTW
n_timesteps = 7  # 7-day window
n_features = len(feature_cols) // n_timesteps
X = k_mediods_data[feature_cols].values.reshape(-1, n_timesteps, n_features)

# Computing DTW Distance Matrix
print("Computing DTW matrix (this may take a bit more time)...")
dtw_matrix = cdist_dtw(X, n_jobs=-1)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute DTW matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
k_values = range(2, 16)

# Preparing feature matrix for Calinski-Harabasz and Davies-Bouldin
X_features = k_mediods_data[feature_cols].values

for k in k_values:
    print(f"\nClustering with k={k}...")
    kmedoids = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
    labels = kmedoids.fit_predict(dtw_matrix)

    # Computing metrics
    metrics = {
        'k': k,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (uses DTW distance matrix)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(dtw_matrix, labels, metric="precomputed")

    # Calinski-Harabasz and Davies-Bouldin (use feature space)
    if len(np.unique(labels)) > 1:
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_features, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_features, labels)
        except Exception as e:
            print(f"Error computing metrics: {e}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Plotting All Metrics
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(k_values, metrics_df['silhouette'], marker='o', color='red')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.grid(True)

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(k_values, metrics_df['calinski_harabasz'], marker='o', color='blue')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Calinski-Harabasz Index")
plt.title("Calinski-Harabasz Index")
plt.grid(True)

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(k_values, metrics_df['davies_bouldin'], marker='o', color='green')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Davies-Bouldin Index")
plt.title("Davies-Bouldin Index")
plt.grid(True)

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette   score metric.")

        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['k'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['k'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['k'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': k_mediods_data['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
        continue
    for k in clusters:
        print(f"\nTesting k={k} (from {metric} metric)")
        kmedoids = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
        labels = kmedoids.fit_predict(dtw_matrix)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for k={k} ({metric}):")
        display(results)







#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ==============================================================
#    Agglomerative Clustering (Ward) using 'euclidean' metric
# ==============================================================

# Creating a copy of final pre_processed dataframe to use it for Agglomerative Clustering (Ward) clustering.
agglomerative_data=preprocessed_data.copy()
print(agglomerative_data.shape)
agglomerative_data.head()

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Loading preprocessed data
feature_cols = [col for col in agglomerative_data.columns if col not in ['hadm_id', 'subject_id']]
X = agglomerative_data[feature_cols].values

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = list(range(2, 16))

for k in range_n_clusters:
    print(f"Clustering with k={k}...")

    # Ward Agglomerative Clustering
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage='ward',
        metric='euclidean'
    )
    labels = model.fit_predict(X)

    # Computing metrics
    metrics = {
        'k': k,
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels)
    }
    metrics_results.append(metrics)

# Converting it into a DataFrame
metrics_df = pd.DataFrame(metrics_results)

# Plotting Scores
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o')
plt.grid()
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='orange')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters (k)')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters (k)')

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette   score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['k'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['k'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['k'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters

def run_fisher_test(labels, outcome_data):
    # Merging cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': agglomerative_data['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='inner'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
        continue
    for k in clusters:
        print(f"\nTesting k={k} (from {metric} metric)")
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage='ward',
            metric='euclidean'
        )
        labels = model.fit_predict(X)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for k={k} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ================================================================================
#    Agglomerative Clustering using Manhattan Distance matrix (without Imputation)
# ================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import time

# Calculating time to compute custom Manhattan distance matrix
start = time.time()

print("Computing Manhattan distance matrix...")

# Computing Manhattan distance using only overlapping non-missing features for each pair
def custom_manhattan_distance(X):
    n = X.shape[0]
    distance_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i+1, n):
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])
            if mask.sum() > 0:
                # Manhattan distance: sum of absolute differences
                dist = np.sum(np.abs(X[i, mask] - X[j, mask]))
                distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix

manhattan_dist = custom_manhattan_distance(feature_matrix)

# Preparing distance matrix for clustering
max_dist = np.nanmax(manhattan_dist)
manhattan_dist_filled = np.where(np.isnan(manhattan_dist), max_dist + 1, manhattan_dist)
np.fill_diagonal(manhattan_dist_filled, 0)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Manhattan distance matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)

for n_clusters in range_n_clusters:
    print(f"Clustering with {n_clusters} clusters...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(manhattan_dist_filled)  # Use Manhattan distance

    # Computing metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (uses custom distance matrix)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(manhattan_dist_filled, labels, metric='precomputed')

    # Since Calinski-Harabasz and Davies-Bouldin require feature space
    if len(np.unique(labels)) > 1:
        try:
            # Prepare preprocessed feature matrix (imputed+normalized) for Calinski/Davies
            X_preprocessed = preprocessed_data[feature_cols].values
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_preprocessed, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_preprocessed, labels)
        except Exception as e:
            print(f"Error computing metrics: {e}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Plotting All Metrics
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o')
plt.grid()
plt.title('Silhouette Score (Manhattan)')
plt.xlabel('Number of Clusters')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='orange')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters')

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': pivot['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
        continue
    for n in clusters:
        print(f"\nTesting n_clusters={n} (from {metric} metric)")
        clustering = AgglomerativeClustering(
            n_clusters=n,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(manhattan_dist_filled)  # Use Manhattan distance
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for n={n} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ================================================================================
#    Agglomerative Clustering using Mahalanobis Distance matrix (without Imputation)
# ================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import time
from numpy.linalg import inv, LinAlgError
from scipy.linalg import pinv

# Calculating time to compute Mahalanobis distance matrix
start = time.time()

print("Computing Mahalanobis distance matrix...")

# Precompute global covariance matrix using pairwise complete observations
print("Computing global covariance matrix...")
df = pd.DataFrame(feature_matrix)
cov_global = df.cov().values  # Uses pairwise complete observations

def custom_mahalanobis_distance(X, global_cov):
    n = X.shape[0]
    distance_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i+1, n):
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])
            n_overlap = mask.sum()

            if n_overlap > 0:
                x_i = X[i, mask]
                x_j = X[j, mask]
                diff = x_i - x_j

                # Extract sub-covariance matrix for overlapping features
                sub_cov = global_cov[mask][:, mask]

                # Handle small covariance matrices
                if n_overlap == 1:
                    # For single feature, use normalized absolute difference
                    dist = np.abs(diff[0]) / np.sqrt(sub_cov[0,0]) if sub_cov[0,0] > 0 else np.abs(diff[0])
                else:
                    try:
                        # Attempt to invert covariance matrix
                        inv_cov = inv(sub_cov)
                    except LinAlgError:
                        # Use pseudo-inverse if matrix is singular
                        inv_cov = pinv(sub_cov)

                    # Compute Mahalanobis distance
                    dist = np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))

                distance_matrix[i, j] = distance_matrix[j, i] = dist

    return distance_matrix

# Compute Mahalanobis distance matrix
mahalanobis_dist = custom_mahalanobis_distance(feature_matrix, cov_global)

# Preparing distance matrix for clustering
max_dist = np.nanmax(mahalanobis_dist)
mahalanobis_dist_filled = np.where(np.isnan(mahalanobis_dist), max_dist + 1, mahalanobis_dist)
np.fill_diagonal(mahalanobis_dist_filled, 0)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Mahalanobis distance matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)

for n_clusters in range_n_clusters:
    print(f"Clustering with {n_clusters} clusters...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(mahalanobis_dist_filled)

    # Computing metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (uses custom distance matrix)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(mahalanobis_dist_filled, labels, metric='precomputed')

    # Calinski-Harabasz and Davies-Bouldin require feature space
    if len(np.unique(labels)) > 1:
        try:
            # Prepare preprocessed feature matrix (imputed+normalized)
            X_preprocessed = preprocessed_data[feature_cols].values
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_preprocessed, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_preprocessed, labels)
        except Exception as e:
            print(f"Error computing metrics: {e}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Plotting All Metrics
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o')
plt.grid()
plt.title('Silhouette Score (Mahalanobis)')
plt.xlabel('Number of Clusters')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='orange')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters')

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': pivot['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
        continue
    for n in clusters:
        print(f"\nTesting n_clusters={n} (from {metric} metric)")
        clustering = AgglomerativeClustering(
            n_clusters=n,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(mahalanobis_dist_filled)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for n={n} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ================================================================================
#    Agglomerative Clustering using Cosine Distance matrix  (without Imputation)
# ================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import time

# Calculating time to compute Cosine distance matrix
start = time.time()

print("Computing Cosine distance matrix...")

def custom_cosine_distance(X):
    """
    Compute cosine distance using only overlapping non-missing features
    """
    n = X.shape[0]
    distance_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i+1, n):
            # Find overlapping non-missing features
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])

            if mask.sum() > 0:
                # Extract vectors for overlapping features
                vec_i = X[i, mask]
                vec_j = X[j, mask]

                # Compute cosine distance (1 - cosine similarity)
                dot_product = np.dot(vec_i, vec_j)
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)

                if norm_i > 0 and norm_j > 0:
                    cosine_sim = dot_product / (norm_i * norm_j)
                    # Clip cosine similarity to [-1, 1] to handle numerical precision
                    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                    cosine_dist = 1 - cosine_sim
                    # Ensure distance is non-negative
                    distance_matrix[i, j] = distance_matrix[j, i] = max(0.0, cosine_dist)
                else:
                    # Handle zero-norm vectors (all values zero)
                    distance_matrix[i, j] = distance_matrix[j, i] = 1.0

    return distance_matrix

# Compute Cosine distance matrix
cosine_dist = custom_cosine_distance(feature_matrix)

# Preparing distance matrix for clustering
max_dist = np.nanmax(cosine_dist)
cosine_dist_filled = np.where(np.isnan(cosine_dist), max_dist + 1, cosine_dist)
np.fill_diagonal(cosine_dist_filled, 0)

# **Additional safety check: Ensure all values are non-negative**
cosine_dist_filled = np.abs(cosine_dist_filled)  # Force all values to be non-negative

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Cosine distance matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)

for n_clusters in range_n_clusters:
    print(f"Clustering with {n_clusters} clusters...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(cosine_dist_filled)

    # Computing metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (uses custom distance matrix)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(cosine_dist_filled, labels, metric='precomputed')

    # Calinski-Harabasz and Davies-Bouldin require feature space
    if len(np.unique(labels)) > 1:
        try:
            # Prepare preprocessed feature matrix (imputed+normalized)
            X_preprocessed = preprocessed_data[feature_cols].values
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_preprocessed, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_preprocessed, labels)
        except Exception as e:
            print(f"Error computing metrics: {e}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Plotting All Metrics
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o')
plt.grid()
plt.title('Silhouette Score (Cosine)')
plt.xlabel('Number of Clusters')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='orange')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters')

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': pivot['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
        continue
    for n in clusters:
        print(f"\nTesting n_clusters={n} (from {metric} metric)")
        clustering = AgglomerativeClustering(
            n_clusters=n,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(cosine_dist_filled)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for n={n} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ================================================================================
#    Agglomerative Clustering using Euclidean Distance matrix (without Imputation)
# ================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import time

# Calculating time to compute custom (Euclidean) distance matrix
start = time.time()

'''
Using Feature Matrix "feature_matrix" that we created above before applying Normalization/Imputation i.e

feature_cols = [col for col in pivot.columns if col not in ['hadm_id', 'subject_id']]
feature_matrix = pivot[feature_cols].values

'''
print("Computing Euclidean distance matrix...")

# Computing Euclidean distance using only overlapping non-missing features for each pair
def custom_euclidean_distance(X):
    n = X.shape[0]
    distance_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i+1, n):
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])
            if mask.sum() > 0:
                dist = np.linalg.norm(X[i, mask] - X[j, mask])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix

euclidean_dist = custom_euclidean_distance(feature_matrix)

# Preparing distance matrix for clustering
max_dist = np.nanmax(euclidean_dist)
euclidean_dist_filled = np.where(np.isnan(euclidean_dist), max_dist + 1, euclidean_dist)
np.fill_diagonal(euclidean_dist_filled, 0)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Euclidean distance matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)

for n_clusters in range_n_clusters:
    print(f"Clustering with {n_clusters} clusters...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(euclidean_dist_filled)

    # Computing metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (uses custom distance matrix)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(euclidean_dist_filled, labels, metric='precomputed')

    # Since Calinski-Harabasz and Davies-Bouldin require feature space so We'll use the imputed/normalized version from preprocessed data
    if len(np.unique(labels)) > 1:
        try:
            # Prepare preprocessed feature matrix (imputed+normalized) for Calinski/Davies
            X_preprocessed = preprocessed_data[feature_cols].values

            # Using (imputed+normalized) feature coloumns
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_preprocessed, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_preprocessed, labels)
        except Exception as e:
            print(f"Error computing metrics: {e}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Plotting All Metrics
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o')
plt.grid()
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='orange')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters')

plt.tight_layout()
plt.show()

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette   score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop-3 cluster numbers per metric:")
print(top_clusters)

# Fisher’s Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': pivot['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
        continue
    for n in clusters:
        print(f"\nTesting n_clusters={n} (from {metric} metric)")
        clustering = AgglomerativeClustering(
            n_clusters=n,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(euclidean_dist_filled)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for n={n} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ====================================================================
#    Agglomerative Clustering using DTW matrix (fast, dtaidistance)
# ====================================================================

# Creating a copy of final pre_processed dataframe to use it for Agglomerative Clustering.
agglomerative_data=preprocessed_data.copy()
print(agglomerative_data.shape)
agglomerative_data.head()

#!pip install dtaidistance

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import time

# Preparing data
feature_cols = [col for col in agglomerative_data.columns if col not in ['hadm_id', 'subject_id']]

# Reshape for DTW: (n_samples, n_timesteps, n_features)
n_timesteps = 7
n_features = len(feature_cols) // n_timesteps
X = agglomerative_data[feature_cols].values.reshape(-1, n_timesteps, n_features)

# Flatten each time series for univariate DTW
series_list = [x.flatten() for x in X]

# Computing DTW distance matrix (fast, parallelized)
print("Computing DTW matrix with dtaidistance (fast)...")
start = time.time()
dtw_matrix = dtw.distance_matrix_fast(
    series_list,
    parallel=True,
    use_c=True,
)
np.fill_diagonal(dtw_matrix, 0)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute DTW matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)
X_features = agglomerative_data[feature_cols].values

for k in range_n_clusters:
    print(f"Clustering with k={k}...")
    model = AgglomerativeClustering(
        n_clusters=k,
        metric='precomputed',
        linkage='average'
    )
    labels = model.fit_predict(dtw_matrix)

    metrics = {
        'k': k,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(dtw_matrix, labels, metric="precomputed")
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_features, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_features, labels)
        except Exception as e:
            print(f"Error computing metrics: {e}")
    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Find best silhouette score and cluster number
best_sil_idx = metrics_df['silhouette'].idxmax()
best_sil_k = metrics_df.loc[best_sil_idx, 'k']
best_sil_score = metrics_df.loc[best_sil_idx, 'silhouette']

# Plotting All Metrics
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o')
plt.grid()
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.legend()

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='orange')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters (k)')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters (k)')
plt.tight_layout()
plt.show()

print(f'Best Silhouette: k={best_sil_k} (Score: {best_sil_score:.3f})')

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has value above threshold {silhouette_threshold}.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['k'].tolist()
    top_clusters['silhouette'] = top_sil
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['k'].tolist()
    top_clusters['calinski_harabasz'] = top_ch
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['k'].tolist()
    top_clusters['davies_bouldin'] = top_db
    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop-3 cluster numbers per metric:")
print(top_clusters)

# Fisher's Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    merged_data = pd.DataFrame({
        'hadm_id': agglomerative_data['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='inner'
    ).dropna(subset=['target'])
    p_values = []
    clusters = np.unique(labels)
    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']
        a = (target_in_cluster == 1).sum()
        b = (target_out_cluster == 1).sum()
        c = (target_in_cluster == 0).sum()
        d = (target_out_cluster == 0).sum()
        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })
    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
        continue
    for k in clusters:
        print(f"\nTesting k={k} (from {metric} metric)")
        model = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(dtw_matrix)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for k={k} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ===================================================
#     Agglomerative Clustering using DTW matrix
# ===================================================

# Creating a copy of final pre_processed dataframe to use it for Agglomerative Clustering.
agglomerative_data=preprocessed_data.copy()
print(agglomerative_data.shape)
agglomerative_data.head()

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Calculating time to compute DTW matrix
start = time.time()

# Loading preprocessed data (features)
feature_cols = [col for col in agglomerative_data.columns if col not in ['hadm_id', 'subject_id']]

# Preparing Data for DTW
n_timesteps = 7  # 7-day window
n_features = len(feature_cols) // n_timesteps
X = agglomerative_data[feature_cols].values.reshape(-1, n_timesteps, n_features)

# Computing DTW Distance Matrix (DTW matrix is already computed above in K-Mediods section)
print("Computing DTW matrix (this may take a bit more time)...")
dtw_matrix = cdist_dtw(X, n_jobs=-1)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute DTW matrix: {minutes} minutes and {seconds} seconds")

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
range_n_clusters = range(2, 16)

# Prepare feature matrix for Calinski-Harabasz and Davies-Bouldin
X_features = agglomerative_data[feature_cols].values

for k in range_n_clusters:
    print(f"Clustering with k={k}...")

    # Agglomerative Clustering with DTW matrix (fixed parameter)
    model = AgglomerativeClustering(
        n_clusters=k,
        metric='precomputed',  # Updated from 'affinity'
        linkage='average'
    )
    labels = model.fit_predict(dtw_matrix)

    # Compute metrics
    metrics = {
        'k': k,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    # Silhouette (uses DTW matrix)
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(dtw_matrix, labels, metric="precomputed")

    # Calinski-Harabasz and Davies-Bouldin (use feature space)
    if len(np.unique(labels)) > 1:
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_features, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X_features, labels)
        except Exception as e:
            print(f"Error computing metrics: {e}")

    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)

# Finding Best Silhouette score alongwith its cluster number
best_sil_idx = metrics_df['silhouette'].idxmax()
best_sil_k = metrics_df.loc[best_sil_idx, 'k']
best_sil_score = metrics_df.loc[best_sil_idx, 'silhouette']

# Plotting All Metrics
plt.figure(figsize=(15, 5))

# Silhouette Score with best cluster highlighted
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, metrics_df['silhouette'], marker='o')
plt.grid()
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.legend()

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], marker='o', color='orange')
plt.grid()
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters (k)')

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, metrics_df['davies_bouldin'], marker='o', color='green')
plt.grid()
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters (k)')

plt.tight_layout()
plt.show()

print(f'Best Silhouette: k={best_sil_k} (Score: {best_sil_score:.3f})')

metrics_df

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
        print(f"Silhouette Score Analysis: No cluster has more value than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore no cluster has been selected Silhouette score metric.")
        top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['k'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['k'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['k'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop-3 cluster numbers per metric:")
print(top_clusters)

# Fisher's Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': agglomerative_data['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='inner'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
        continue
    for k in clusters:
        print(f"\nTesting k={k} (from {metric} metric)")
        model = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(dtw_matrix)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for k={k} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ===============================================================================
#     Agglomerative Clustering using Binary Distance Matrices (Threshold= 1%)
# ===============================================================================

'''
When we create a binary dataset, we do not need to perform Normalization/Imputation therefore to compute Clustering using Binary matrix,
we'll use unimputed measured labs i.e "filtered_common_labs_one_percent" dataframe (where we used "Threshold= 1%") which we  created during
"Pre-processing" before we applied Normalization/Imputation.

'''
#Importing required libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import pairwise_distances

# Calculating time to compute Binary distance matrix
start = time.time()

# Creating binary dataset while using the labs measured with a "1%" threshold.
binary_pivot = filtered_common_labs_one_percent.copy()
binary_pivot['measured'] = 1
binary_dataset = binary_pivot.pivot_table(
    index='hadm_id',
    columns='itemid',
    values='measured',
    aggfunc='max',
    fill_value=0
).reset_index()

# Preparing features
binary_features = binary_dataset.drop(columns=['hadm_id'])
binary_features_bool = binary_features.astype(bool)
binary_features_np = binary_features.values
binary_features_bool_np = binary_features_bool.values

# Computing Distance matrices
print("Computing 'Hamming', 'Jaccard' & 'Dice' Distance matrices...")
start_dist = time.time()

hamming_dist = pairwise_distances(binary_features_np, metric='hamming')
jaccard_dist = pairwise_distances(binary_features_bool_np, metric='jaccard')

def dice_distance(X):
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A = X[i]
            B = X[j]
            intersection = np.sum(A & B)
            size_sum = np.sum(A) + np.sum(B)
            dist = 0.0 if size_sum == 0 else 1 - (2 * intersection / size_sum)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

dice_dist = dice_distance(binary_features_bool_np)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Distance matrices: {minutes} minutes and {seconds} seconds")

# Metric Tracking for All Distance Matrices
metrics_results = {
    'Hamming': [],
    'Jaccard': [],
    'Dice': []
}
range_n_clusters = range(2, 16)

for metric_name, dist_matrix in zip(['Hamming', 'Jaccard', 'Dice'],
                                    [hamming_dist, jaccard_dist, dice_dist]):
    print(f"\nProcessing {metric_name} distance matrix...")
    for n_clusters in range_n_clusters:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(dist_matrix)

        # Computing metrics
        metrics = {
            'n_clusters': n_clusters,
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan
        }

        # Silhouette (uses distance matrix)
        if len(np.unique(labels)) > 1:
            metrics['silhouette'] = silhouette_score(dist_matrix, labels, metric='precomputed')

        # Calinski-Harabasz and Davies-Bouldin (use feature space)
        if len(np.unique(labels)) > 1:
            try:
                metrics['calinski_harabasz'] = calinski_harabasz_score(binary_features_np, labels)
                metrics['davies_bouldin'] = davies_bouldin_score(binary_features_np, labels)
            except Exception as e:
                print(f"  Metric computation error: {str(e)}")

        metrics_results[metric_name].append(metrics)

# Convert to DataFrames
metrics_dfs = {k: pd.DataFrame(v) for k, v in metrics_results.items()}

# Plotting All Metrics
def plot_metrics(metrics_df, metric_name):
    plt.figure(figsize=(15, 5))

    # Silhouette Score
    plt.subplot(1, 3, 1)
    plt.plot(range_n_clusters, metrics_df['silhouette'], 'bo-')
    plt.title(f'Silhouette Score ({metric_name})')
    plt.xlabel('Number of Clusters')
    plt.grid(True)

    # Calinski-Harabasz Index
    plt.subplot(1, 3, 2)
    plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], 'ro-')
    plt.title(f'Calinski-Harabasz Index ({metric_name})')
    plt.xlabel('Number of Clusters')
    plt.grid(True)

    # Davies-Bouldin Index
    plt.subplot(1, 3, 3)
    plt.plot(range_n_clusters, metrics_df['davies_bouldin'], 'go-')
    plt.title(f'Davies-Bouldin Index ({metric_name})')
    plt.xlabel('Number of Clusters')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

for metric_name, df in metrics_dfs.items():
    plot_metrics(df, metric_name)

# Top Cluster Selection with Threshold Check
def get_top_clusters(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist() if len(sil_df) >= 3 else []
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

# Printing all metrics for all distance matrices
for metric_name, df in metrics_dfs.items():
    print(f"\n{'='*30}\nMetrics for {metric_name} distance\n{'='*30}")
    display(df[['n_clusters', 'silhouette', 'davies_bouldin', 'calinski_harabasz']])

# Top-3 Cluster Numbers for Each Binary Distance Metric

print("\n" + "="*60)
print("TOP-3 CLUSTER NUMBERS FOR BINARY DISTANCE METRICS")
print("="*60)

for metric_name, df in metrics_dfs.items():
    top_clusters = get_top_clusters(df, silhouette_threshold=0.25)

    print(f"\n🔸 {metric_name} Distance Matrix:")
    print(f"Top-3 cluster numbers per metric:")
    print(f"{top_clusters}")

    # Check if any silhouette clusters meet threshold
    if not top_clusters['silhouette']:
        print(f"⚠️  Note: No clusters with Silhouette > 0.25 for {metric_name}")

print("\n" + "="*60)

# Fisher's Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    merged_data = pd.DataFrame({
        'hadm_id': binary_dataset['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in = merged_data.loc[in_cluster, 'target']
        target_out = merged_data.loc[~in_cluster, 'target']

        a = (target_in == 1).sum()  # Target=1 in cluster
        b = (target_out == 1).sum() # Target=1 outside cluster
        c = (target_in == 0).sum()  # Target=0 in cluster
        d = (target_out == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    return pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric_name, df in metrics_dfs.items():
    top_clusters = get_top_clusters(df)
    print(f"\nTop clusters for {metric_name}:")
    print(top_clusters)

    for metric_type, clusters in top_clusters.items():
        if not clusters:
            print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
            continue

        for n in clusters:
            print('========================================================================================')
            print(f"\nTesting {metric_name} with {n} clusters (from {metric_type} metric)")
            model = AgglomerativeClustering(
                n_clusters=n,
                metric='precomputed',
                linkage='average'
            )
            dist_matrix = eval(f"{metric_name.lower()}_dist")
            labels = model.fit_predict(dist_matrix)
            results = run_fisher_test(labels, cohort_data)
            print(f"Results for {metric_name} (k={n}, {metric_type}):")
            display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# =============================================================================
#     Agglomerative Clustering using Binary Distance Matrices (Threshold= 75%)
# =============================================================================

'''
When we create a binary dataset, we do not need to perform Normalization/Imputation therefore to compute Clustering using Binary matrix,
we'll use unimputed measured labs i.e "filtered_common_labs" dataframe (where we used "Threshold= 75%") which we created during "Pre-processing"
before we applied Normalization/Imputation.

'''
#Importing required libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import pairwise_distances

#Calculating time to compute Binary distance matrix
start = time.time()

# Creating binary dataset
binary_pivot = filtered_common_labs.copy()
binary_pivot['measured'] = 1
binary_dataset = binary_pivot.pivot_table(
    index='hadm_id',
    columns='itemid',
    values='measured',
    aggfunc='max',
    fill_value=0
).reset_index()

# Preparing features
binary_features = binary_dataset.drop(columns=['hadm_id'])
binary_features_bool = binary_features.astype(bool)
binary_features_np = binary_features.values
binary_features_bool_np = binary_features_bool.values

# Computing Distance matrices
print("Computing 'Hamming', 'Jaccard' & 'Dice' Distance matrices...")
start_dist = time.time()

hamming_dist = pairwise_distances(binary_features_np, metric='hamming')
jaccard_dist = pairwise_distances(binary_features_bool_np, metric='jaccard')

def dice_distance(X):
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A = X[i]
            B = X[j]
            intersection = np.sum(A & B)
            size_sum = np.sum(A) + np.sum(B)
            dist = 0.0 if size_sum == 0 else 1 - (2 * intersection / size_sum)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

dice_dist = dice_distance(binary_features_bool_np)

elapsed_time = time.time() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Execution time to compute Distance matrices: {minutes} minutes and {seconds} seconds")

# Metric Tracking for All Distance Matrices
metrics_results = {
    'Hamming': [],
    'Jaccard': [],
    'Dice': []
}
range_n_clusters = range(2, 16)

for metric_name, dist_matrix in zip(['Hamming', 'Jaccard', 'Dice'],
                                    [hamming_dist, jaccard_dist, dice_dist]):
    print(f"\nProcessing {metric_name} distance matrix...")
    for n_clusters in range_n_clusters:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(dist_matrix)

        # Computing metrics
        metrics = {
            'n_clusters': n_clusters,
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan
        }

        # Silhouette (uses distance matrix)
        if len(np.unique(labels)) > 1:
            metrics['silhouette'] = silhouette_score(dist_matrix, labels, metric='precomputed')

        # Calinski-Harabasz and Davies-Bouldin (use feature space)
        if len(np.unique(labels)) > 1:
            try:
                metrics['calinski_harabasz'] = calinski_harabasz_score(binary_features_np, labels)
                metrics['davies_bouldin'] = davies_bouldin_score(binary_features_np, labels)
            except Exception as e:
                print(f"  Metric computation error: {str(e)}")

        metrics_results[metric_name].append(metrics)

# Convert to DataFrames
metrics_dfs = {k: pd.DataFrame(v) for k, v in metrics_results.items()}

# Plotting All Metrics
def plot_metrics(metrics_df, metric_name):
    plt.figure(figsize=(15, 5))

    # Silhouette Score
    plt.subplot(1, 3, 1)
    plt.plot(range_n_clusters, metrics_df['silhouette'], 'bo-')
    plt.title(f'Silhouette Score ({metric_name})')
    plt.xlabel('Number of Clusters')
    plt.grid(True)

    # Calinski-Harabasz Index
    plt.subplot(1, 3, 2)
    plt.plot(range_n_clusters, metrics_df['calinski_harabasz'], 'ro-')
    plt.title(f'Calinski-Harabasz Index ({metric_name})')
    plt.xlabel('Number of Clusters')
    plt.grid(True)

    # Davies-Bouldin Index
    plt.subplot(1, 3, 3)
    plt.plot(range_n_clusters, metrics_df['davies_bouldin'], 'go-')
    plt.title(f'Davies-Bouldin Index ({metric_name})')
    plt.xlabel('Number of Clusters')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

for metric_name, df in metrics_dfs.items():
    plot_metrics(df, metric_name)

# Top Cluster Selection with Threshold Check
def get_top_clusters(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    top_sil = sil_df.nlargest(3, 'silhouette')['n_clusters'].tolist() if len(sil_df) >= 3 else []
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['n_clusters'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['n_clusters'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

# Printing all metrics for all distance matrices
for metric_name, df in metrics_dfs.items():
    print(f"\n{'='*30}\nMetrics for {metric_name} distance\n{'='*30}")
    display(df[['n_clusters', 'silhouette', 'davies_bouldin', 'calinski_harabasz']])

# Top-3 Cluster Numbers for Each Binary Distance Metric

print("\n" + "="*60)
print("TOP-3 CLUSTER NUMBERS FOR BINARY DISTANCE METRICS")
print("="*60)

for metric_name, df in metrics_dfs.items():
    top_clusters = get_top_clusters(df, silhouette_threshold=0.25)

    print(f"\n🔸 {metric_name} Distance Matrix:")
    print(f"Top-3 cluster numbers per metric:")
    print(f"{top_clusters}")

    # Check if any silhouette clusters meet threshold
    if not top_clusters['silhouette']:
        print(f"⚠️  Note: No clusters with Silhouette > 0.25 for {metric_name}")

print("\n" + "="*60)

# Fisher's Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    merged_data = pd.DataFrame({
        'hadm_id': binary_dataset['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='left'
    ).dropna(subset=['target'])

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in = merged_data.loc[in_cluster, 'target']
        target_out = merged_data.loc[~in_cluster, 'target']

        a = (target_in == 1).sum()  # Target=1 in cluster
        b = (target_out == 1).sum() # Target=1 outside cluster
        c = (target_in == 0).sum()  # Target=0 in cluster
        d = (target_out == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    return pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

print("\nRunning Fisher's tests on Top-3 clusters...")
for metric_name, df in metrics_dfs.items():
    top_clusters = get_top_clusters(df)
    print(f"\nTop clusters for {metric_name}:")
    print(top_clusters)

    for metric_type, clusters in top_clusters.items():
        if not clusters:
            print(f"No cluster counts selected for {metric}. Therefore No Fisher's test will be conducted for {metric}.")
            continue

        for n in clusters:
            print('========================================================================================')
            print(f"\nTesting {metric_name} with {n} clusters (from {metric_type} metric)")
            model = AgglomerativeClustering(
                n_clusters=n,
                metric='precomputed',
                linkage='average'
            )
            dist_matrix = eval(f"{metric_name.lower()}_dist")
            labels = model.fit_predict(dist_matrix)
            results = run_fisher_test(labels, cohort_data)
            print(f"Results for {metric_name} (k={n}, {metric_type}):")
            display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ==============================================
#   DBSCAN clustering using 'euclidean' metric
# ==============================================

# Creating a copy of final pre_processed dataframe to use it for DBSCAN clustering.
DBscan_data=preprocessed_data.copy()
print(DBscan_data.shape)
DBscan_data.head()

# Importing necessary libraries
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Preparing data
feature_cols = [col for col in DBscan_data.columns if col not in ['hadm_id', 'subject_id']]
X = DBscan_data[feature_cols].values

# Finding Optimal ε (eps) for DBSCAN by computing k-distance graph (Elbow method)
neighbors = NearestNeighbors(n_neighbors=5)  # min_samples=5 as starting point
neighbors_fit = neighbors.fit(X)
distances, _ = neighbors_fit.kneighbors(X)

# Sort distances and plot
distances = np.sort(distances[:, -1])
plt.figure(figsize=(6, 4))
plt.plot(distances)
plt.xlabel('Samples')
plt.ylabel('5th Nearest Neighbor Distance')
plt.title('Elbow Method for Optimal ε')
plt.grid()
plt.show()

# Parameters (based on k-distance plot)
eps = 20    # appears to be somewhere between 15-25
min_samples = 5

# DBSCAN with Euclidean distance
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
cluster_labels = dbscan.fit_predict(X)

# Add labels to DataFrame
DBscan_data['dbscan_cluster'] = cluster_labels

# Exclude noise points (label=-1) for metrics
valid_labels = cluster_labels != -1
X_valid = X[valid_labels]
labels_valid = cluster_labels[valid_labels]

if len(np.unique(labels_valid)) > 1:  # Need to have at least 2 clusters for metrics otherwise they become meaningless.
    silhouette = silhouette_score(X_valid, labels_valid)
    calinski = calinski_harabasz_score(X_valid, labels_valid)
    davies = davies_bouldin_score(X_valid, labels_valid)
else:
    silhouette = calinski = davies = np.nan

print(f"Silhouette Score: {silhouette:.3f}   \nCalinski-Harabasz Index: {calinski:.3f}   \nDavies-Bouldin Index: {davies:.3f}")
print(f"Number of clusters: {len(np.unique(labels_valid))}")
print(f"Noise points: {sum(cluster_labels == -1)}")

# Merge with target variable
merged_data = DBscan_data.merge(
    cohort_data[['hadm_id', 'target']],
    on='hadm_id',
    how='inner'
)

# Checking for missing targets
if merged_data['target'].isnull().any():
    print(f"Warning: {merged_data['target'].isnull().sum()} admissions have missing target values")
    merged_data = merged_data.dropna(subset=['target'])

# Testing cluster-target association (while excluding noise)
clusters = np.unique(cluster_labels)
clusters = clusters[clusters != -1]  # Noise is excluded
pvals = []

for cluster in clusters:
    in_cluster = merged_data['dbscan_cluster'] == cluster
    target_in = merged_data.loc[in_cluster, 'target']
    target_out = merged_data.loc[~in_cluster, 'target']

    a = (target_in == 1).sum()
    b = (target_out == 1).sum()
    c = (target_in == 0).sum()
    d = (target_out == 0).sum()

    _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
    pvals.append(p)

# Correct for multiple testing
rejected, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')

# Results
results = pd.DataFrame({
    'Cluster': clusters,
    'Raw_p_value': pvals,
    'Corrected_p_value': pvals_corrected,
    'Significant_after_correction': rejected
})
print("\nCluster-Target Association Results for DBSCAN clustering using euclidean metric:")
display(results.sort_values('Corrected_p_value'))


# Interpretation
significant_clusters = results[results['Significant_after_correction']]
if not significant_clusters.empty:
    print(f"\nSignificant clusters (α=0.05): {significant_clusters['Cluster'].tolist()}")
else:
    print("\nConclusion: No significant clusters found after correction.")









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# ========================================
#    DBSCAN clustering using DTW matrix
# ========================================

# Creating a copy of final pre_processed dataframe to use it for DBSCAN clustering.
DBscan_data=preprocessed_data.copy()
print(DBscan_data.shape)
DBscan_data.head()

# Importing necessary libraries
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tslearn.metrics import cdist_dtw
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Loading preprocessed data (features)
feature_cols = [col for col in DBscan_data.columns if col not in ['hadm_id', 'subject_id']]

# # Preparing Data for DTW
# n_timesteps = 14  # 14-day window
# n_features = len(feature_cols) // n_timesteps
# X = DBscan_data[feature_cols].values.reshape(-1, n_timesteps, n_features)

# # Computing DTW Distance Matrix (DTW matrix is already computed above in K-Mediods section)
# print("Computing DTW matrix (this may take a bit more time)...")
# dtw_matrix = cdist_dtw(X, n_jobs=-1)

# Finding Optimal ε (eps) for DBSCAN by computing k-distance graph
def plot_k_distance(distance_matrix, k):
    neighbors = NearestNeighbors(n_neighbors=k, metric='precomputed')
    neighbors.fit(distance_matrix)
    distances, _ = neighbors.kneighbors(distance_matrix)
    distances = np.sort(distances[:, -1])

    plt.figure(figsize=(6, 4))
    plt.plot(distances)
    plt.xlabel('Samples')
    plt.ylabel(f'{k}th Nearest Neighbor Distance')
    plt.title('k-Distance Graph for DTW-based DBSCAN')
    plt.grid(True)
    plt.show()

plot_k_distance(dtw_matrix, k=5)

# Parameters (based on k-distance plot)
eps_dtw = 15 # appears to be somewhere between 15-25
min_samples = 10

# Clustering model
dbscan = DBSCAN(
    eps= eps_dtw,
    min_samples= min_samples,
    metric= 'precomputed'  # Using precomputed DTW matrix
)
cluster_labels = dbscan.fit_predict(dtw_matrix)

# Adding labels to DataFrame
DBscan_data['dbscan_cluster'] = cluster_labels

# Evaluating the Clusters
# Exclude noise points (label=-1)
valid_labels = cluster_labels != -1
X_valid = X[valid_labels]
labels_valid = cluster_labels[valid_labels]

if len(np.unique(labels_valid)) > 1:   # Need to have at least 2 clusters for metric.
    silhouette = silhouette_score(dtw_matrix[valid_labels][:, valid_labels], labels_valid, metric="precomputed")
    print(f"Silhouette Score: {silhouette:.3f}")
else:
    print("Not enough clusters for Silhouette Score. Tune your parameters")

print(f"Number of clusters: {len(np.unique(labels_valid))}")
print(f"Noise points: {sum(cluster_labels == -1)}")

# Fisher's Exact Test
# Merging with target variable
merged_data = DBscan_data.merge(
    cohort_data[['hadm_id', 'target']],
    on='hadm_id',
    how='inner'
)

# Checking for missing targets
if merged_data['target'].isnull().any():
    print(f"Warning: {merged_data['target'].isnull().sum()} admissions have missing target values")
    merged_data = merged_data.dropna(subset=['target'])

# Testing cluster-target association (while excluding noise)
clusters = np.unique(cluster_labels)
clusters = clusters[clusters != -1]  # Noise is excluded
pvals = []

for cluster in clusters:
    in_cluster = merged_data['dbscan_cluster'] == cluster
    target_in = merged_data.loc[in_cluster, 'target']
    target_out = merged_data.loc[~in_cluster, 'target']

    a = (target_in == 1).sum()
    b = (target_out == 1).sum()
    c = (target_in == 0).sum()
    d = (target_out == 0).sum()

    _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
    pvals.append(p)

# Correct for multiple testing
rejected, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')

# Results
results = pd.DataFrame({
    'Cluster': clusters,
    'Raw_p_value': pvals,
    'Corrected_p_value': pvals_corrected,
    'Significant_after_correction': rejected
})
print("Cluster-Target Association Results for DBSCAN clustering using DTW matrix:")
display(results.sort_values('Corrected_p_value'))

# Interpretation
significant_clusters = results[results['Significant_after_correction']]
if not significant_clusters.empty:
    print(f"\nSignificant clusters (α=0.05): {significant_clusters['Cluster'].tolist()}")
else:
    print("\nConclusion: No significant clusters found after correction.")









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

# =========================
#    Spectral Clustering
# =========================

# Creating a copy of final pre_processed dataframe to use it for Spectral clustering.
spectral_data=preprocessed_data.copy()
print(spectral_data.shape)
spectral_data.head()

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Load preprocessed data (features)
feature_cols = [col for col in spectral_data.columns if col not in ['hadm_id', 'subject_id']]
X = spectral_data[feature_cols].values

# Prepare Data for HMM
n_timesteps = 7  # 7-day window
n_features = len(feature_cols) // n_timesteps
X_reshaped = X.reshape(-1, n_timesteps, n_features)
n_samples = X_reshaped.shape[0]

# Fit Single HMM to All Data
X_combined = X_reshaped.reshape(-1, n_features)

model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=100,
    verbose=False
)
model.fit(X_combined)

# Extract HMM Features (State Probabilities)
features = []
for seq in X_reshaped:
    logprob, posteriors = model.score_samples(seq)
    features.append(posteriors.mean(axis=0))
features = np.array(features)

# Metric Tracking for Cluster Counts 2-15
metrics_results = []
k_values = range(2, 16)

for k in k_values:
    print(f"\nClustering with k={k}")
    spectral = SpectralClustering(
        n_clusters=k,
        affinity='nearest_neighbors',
        random_state=42
    )
    labels = spectral.fit_predict(features)

    # Compute metrics
    metrics = {
        'k': k,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }

    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(features, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(features, labels)

    metrics_results.append(metrics)
    print(f"Silhouette: {metrics['silhouette']:.4f}, Calinski: {metrics['calinski_harabasz']:.1f}, Davies: {metrics['davies_bouldin']:.4f}")

metrics_df = pd.DataFrame(metrics_results)

# Plotting All Metrics
plt.figure(figsize=(15, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(k_values, metrics_df['silhouette'], marker='o', color='red')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.grid(True)

# Calinski-Harabasz Index
plt.subplot(1, 3, 2)
plt.plot(k_values, metrics_df['calinski_harabasz'], marker='o', color='blue')
plt.title('Calinski-Harabasz Index')
plt.xlabel('Number of Clusters (k)')
plt.grid(True)

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(k_values, metrics_df['davies_bouldin'], marker='o', color='green')
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters (k)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Top Cluster Selection with Threshold Check
def get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25):
    top_clusters = {}

    # Silhouette: Top 3 with score > threshold
    sil_df = metrics_df[metrics_df['silhouette'] > silhouette_threshold]
    if len(sil_df) < 1:
       print(f"Silhouette Score Analysis: No cluster has a value ≥ than pre-defined Threshold, that is '{silhouette_threshold}'. Therefore we have no optimal cluster based on Silhouette   score metric.")
       top_sil = []
    else:
        top_sil = sil_df.nlargest(3, 'silhouette')['k'].tolist()
    top_clusters['silhouette'] = top_sil

    # Calinski-Harabasz: Top 3 highest
    top_ch = metrics_df.nlargest(3, 'calinski_harabasz')['k'].tolist()
    top_clusters['calinski_harabasz'] = top_ch

    # Davies-Bouldin: Top 3 lowest
    top_db = metrics_df.nsmallest(3, 'davies_bouldin')['k'].tolist()
    top_clusters['davies_bouldin'] = top_db

    return top_clusters

top_clusters = get_top_clusters_with_threshold(metrics_df, silhouette_threshold=0.25)
print("\nTop cluster numbers per metric:")
print(top_clusters)

metrics_df

# Fisher's Exact Test on Top Clusters
def run_fisher_test(labels, outcome_data):
    # Merge cluster labels with target outcome
    merged_data = pd.DataFrame({
        'hadm_id': spectral_data['hadm_id'],
        'cluster': labels
    }).merge(
        outcome_data[['hadm_id', 'target']],
        on='hadm_id',
        how='inner'
    )

    p_values = []
    clusters = np.unique(labels)

    for cluster_id in clusters:
        in_cluster = merged_data['cluster'] == cluster_id
        target_in_cluster = merged_data.loc[in_cluster, 'target']
        target_out_cluster = merged_data.loc[~in_cluster, 'target']

        a = (target_in_cluster == 1).sum()  # Target=1 in cluster
        b = (target_out_cluster == 1).sum() # Target=1 outside cluster
        c = (target_in_cluster == 0).sum()  # Target=0 in cluster
        d = (target_out_cluster == 0).sum() # Target=0 outside cluster

        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        p_values.append(p)

    # Multiple Testing Correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # Results Table
    results = pd.DataFrame({
        'Cluster': clusters,
        'Raw_p_value': p_values,
        'Corrected_p_value': pvals_corrected,
        'Significant_after_correction': rejected
    })

    return results

print("\nRunning Fisher's tests on top clusters...")
for metric, clusters in top_clusters.items():
    if not clusters:
        print(f"No cluster counts selected for {metric}. Skipping.")
        continue
    for k in clusters:
        print(f"\nTesting k={k} (from {metric} metric)")
        spectral = SpectralClustering(
            n_clusters=k,
            affinity='nearest_neighbors',
            random_state=42
        )
        labels = spectral.fit_predict(features)
        results = run_fisher_test(labels, cohort_data)
        print(f"Results for k={k} ({metric}):")
        display(results)









#------------------------ x ------------------ x -------------------- x ------------------ x -------------------- x ------------------ x -------------

