from imports import *
import utils
from clustering import *

logger = logging.getLogger("UTILS")

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')

def loading_data(path, col_name, num_row=None):
    if num_row is not None:
        return pd.read_csv(path, parse_dates=[col_name], nrows=num_row)
    else:
        return pd.read_csv(path, parse_dates=[col_name])

def merge_data(data1, data2):
     merged = pd.merge(
        data1,
        data2[['subject_id', 'hadm_id', 'dischtime']],
        on=['subject_id', 'hadm_id'],
        how='inner'
    )
     return merged

def labs_within_n_days_of_discharge(merged, days):
    # Calculating the number of days before discharge
    merged['days_before_discharge'] = (merged['dischtime'] - merged['charttime']).dt.days

    # Filtering for the labs within 7 days before discharge
    filtered_labs = merged[
        (merged['days_before_discharge'] >= 0) &
        (merged['days_before_discharge'] < days)
        ].copy()

    # Selecting required columns
    result = filtered_labs[
        ['itemid', 'valuenum', 'charttime', 'hadm_id', 'subject_id', 'dischtime', 'days_before_discharge']]
    return result

def filtering_labs_by_percentage(result, percentage):
    # Calculating total admissions (hadm_id) and checking how many unique values are there in labs (item_ids)
    unique_labs = result['itemid'].nunique()
    total_admissions = result['hadm_id'].nunique()
    # Counting admissions per itemid
    itemid_counts = result.groupby('itemid')['hadm_id'].nunique().reset_index()
    itemid_counts.rename(columns={'hadm_id': 'admission_count'}, inplace=True)
    # Calculating the threshold i.e 1% of the total admissions,
    threshold = round(percentage * total_admissions)
    logger.info(f"Threshold ({percentage * 100}% of the total admissions): {threshold}\n")

    # Getting common itemids that are conduted by atleast 1% of the patients
    selected_itemids = itemid_counts[itemid_counts['admission_count'] >= threshold]['itemid']

    # Summary
    logger.info(
        f"We have selected {len(selected_itemids)} common lab tests out of total {itemid_counts.shape[0]} which have been conducted by atleast {percentage * 100} % patients.\n")
    logger.info(f"Common itemids ({len(selected_itemids)} lab tests): {list(selected_itemids)}")

    # Filtering the result for required labs only
    filtered_common_labs = result[result['itemid'].isin(selected_itemids)].copy()

    # Counting number of measurements per itemid per admission
    lab_counts = filtered_common_labs.groupby(['itemid', 'hadm_id']).size().reset_index(name='count')

    # For each itemid, calculating the average number of measurements per admission
    avg_counts = lab_counts.groupby('itemid')['count'].mean().reset_index(name='avg_per_admission')

    # Selecting itemids with average >= 2
    selected_itemids_2plus = avg_counts[avg_counts['avg_per_admission'] >= 2]['itemid']

    # Filtering to keep only these labs
    filtered_common_labs = filtered_common_labs[filtered_common_labs['itemid'].isin(selected_itemids_2plus)].copy()

    return filtered_common_labs, unique_labs, total_admissions

def descritization(filtered_common_labs, days):
    # Temporal Discretization (24h Bins)

    # Days should be counted forward from the start of the observation window. This means day 1 is the earliest day in the window, day 7 is the latest.
    filtered_common_labs['day_bin'] = days - filtered_common_labs['days_before_discharge']

    # Aggregating by mean
    agg = (
        filtered_common_labs
        .groupby(['hadm_id', 'itemid', 'day_bin'])['valuenum']
        .mean()
        .reset_index()
    )

    # Creating feature-day column names
    agg['feature_day'] = agg['itemid'].astype(str) + '_day' + agg['day_bin'].astype(str)

    # Creating a vector for each admission and each selected lab feature
    pivot = agg.pivot_table(
        index='hadm_id',
        columns='feature_day',
        values='valuenum'
    )

    # Adding subject_id to vectors to make it more understandable
    pivot = pivot.reset_index().merge(
        filtered_common_labs[['hadm_id', 'subject_id']].drop_duplicates(),
        on='hadm_id',
        how='left'
    )

    # We should be getting a vector of 39(labs) * 7(24 hours/1 day timewindow) = 273 vector length.
    logger.info(
        f" 7-dimensional vector with a vector length of {len(pivot.columns) - 2} is: {pivot.head()}")  # subtracting "2" here as we have two identifiers aswell named "hadm_id" & "subject_id"
    return pivot

def get_feature_matrix(pivot):
    # Feature Matrix
    logger.info(
        "Creating 'Feature Matrix' at this point before applying Normalization/Imputation that we are going to use later for Custom pairwise distance function and then finally using that Distance matrix for Clustering.")

    feature_cols = [col for col in pivot.columns if col not in ['hadm_id', 'subject_id']]
    feature_matrix = pivot[feature_cols].values

    # Converting pivot into long format for proper time-series processing to impute within each admission-lab series
    id_vars = ['hadm_id', 'subject_id']
    feature_cols = [col for col in pivot.columns if col not in id_vars]
    feature_matrix = pivot[feature_cols].values
    return feature_matrix, feature_cols, id_vars

def create_long_format(pivot, feature_cols, id_vars):
    # Converting into long format: hadm_id, subject_id, itemid_day, valuenum
    long_df = pivot.melt(id_vars=id_vars, value_vars=feature_cols,
                         var_name='itemid_day', value_name='valuenum')
    # Extracting itemid and day from column name
    long_df[['itemid', 'day']] = long_df['itemid_day'].str.extract(r'(\d+)_day(\d+)')
    long_df['itemid'] = long_df['itemid'].astype(int)
    long_df['day'] = long_df['day'].astype(int)

    # Computing lab-wise statistics for normalization (Using observed values only)
    lab_stats = long_df.groupby('itemid')['valuenum'].agg(['mean', 'std']).reset_index()
    lab_stats.columns = ['itemid', 'lab_mean', 'lab_std']

    # Merging stats back to main dataframe
    long_df = long_df.merge(lab_stats, on='itemid', how='left')

    # Normalizing before imputation (per lab)
    long_df['valuenum_normalized'] = (long_df['valuenum'] - long_df['lab_mean']) / long_df['lab_std']

    return long_df

def impute_timeseries(group):
    # Sorting by day to maintain temporal order
    group = group.sort_values('day')
    # Linear interpolation (only between existing values)
    group['valuenum_imputed'] = group['valuenum_normalized'].interpolate(method='linear', limit_direction='both')

    return group

def get_imputed_df(long_df, id_vars):
    # Apply to each admission-lab group
    imputed_df = long_df.groupby(['hadm_id', 'itemid']).apply(impute_timeseries).reset_index(drop=True)

    # Converting Pivot back to wide format
    imputed_df['itemid_day'] = imputed_df['itemid'].astype(str) + '_day' + imputed_df['day'].astype(str)
    wide_df = imputed_df.pivot_table(index=id_vars, columns='itemid_day', values='valuenum_imputed').reset_index()

    return imputed_df, wide_df

def knn_impute_and_sort_features(wide_df, id_vars):

    # Identify feature columns (all except identifiers)
    feature_cols = [col for col in wide_df.columns if col not in id_vars]

    # Perform KNN imputation if any missing values exist
    if wide_df[feature_cols].isnull().values.any():
        imputer = KNNImputer(n_neighbors=5)
        wide_df[feature_cols] = imputer.fit_transform(wide_df[feature_cols])

    df = wide_df.copy()

    def sort_key(col_name: str):
        itemid_str, day_str = col_name.split('_day')
        return (int(itemid_str), int(day_str))

    # Sort feature columns by itemid and day
    sorted_feature_cols = sorted(feature_cols, key=sort_key)

    # Reorder DataFrame: identifiers first, then sorted features
    preprocessed_df = df[id_vars + sorted_feature_cols]

    return preprocessed_df


def plot_and_save_kmeans_metrics(metrics_df, filename="kmeans_metrics.png"):
    """
    Plot Silhouette, Calinski-Harabasz, Davies-Bouldin vs k and save to disk.
    """
    
    k_vals = metrics_df['k'].tolist()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(k_vals, metrics_df['silhouette'], marker='o')
    plt.grid()
    plt.title('Silhouette Score')
    plt.xlabel('k')

    plt.subplot(1, 3, 2)
    plt.plot(k_vals, metrics_df['calinski_harabasz'], marker='o')
    plt.grid()
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('k')

    plt.subplot(1, 3, 3)
    plt.plot(k_vals, metrics_df['davies_bouldin'], marker='o')
    plt.grid()
    plt.title('Davies-Bouldin Index')
    plt.xlabel('k')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"KMeans metric plot saved to {filename}")
    
def custom_manhattan_distance(X: np.ndarray) -> np.ndarray:

    n = X.shape[0]
    D = np.full((n, n), np.nan)
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.any():
                D[i, j] = D[j, i] = np.abs(xi[mask] - xj[mask]).sum()
    np.fill_diagonal(D, 0.0)
    return D


def fill_nan_in_distance(D: np.ndarray) -> np.ndarray:

    max_dist = np.nanmax(D)
    D_filled = np.where(np.isnan(D), max_dist + 1, D)
    np.fill_diagonal(D_filled, 0.0)
    return D_filled
    
def plot_and_save_kdist(distances: np.ndarray, filename: str, k_label: str = "k-NN distance"):
    plt.figure(figsize=(6, 4))
    plt.plot(distances)
    plt.xlabel("Samples")
    plt.ylabel(k_label)
    plt.title("Elbow Method for Optimal ε")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
def plot_and_save_kdist_precomputed(distances, filename, k_label="k-NN distance"):
    plt.figure(figsize=(6,4))
    plt.plot(distances)
    plt.xlabel("Samples"); plt.ylabel(k_label)
    plt.title("k-Distance Graph (precomputed)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def run_fisher_loop(
    top_clusters: dict,
    fisher_fn,
    base_kwargs: dict,
    logger: logging.Logger,
    method_label: str
) -> dict:
    """
    Iterate over top_clusters dict and run fisher_fn for each k.
    base_kwargs: common kwargs for fisher_fn (we'll add 'k' each time).
    Returns dict { "<metric>_k=<k>": result_df }
    """
    results = {}
    for metric_name, ks in top_clusters.items():
        if not ks:
            logger.info("No k selected for %s (%s); skipping Fisher.", metric_name, method_label)
            continue
        for k in ks:
            kwargs = base_kwargs.copy()
            kwargs["k"] = k
            res = fisher_fn(**kwargs)
            key = f"{metric_name}_k={k}"
            results[key] = res
            logger.info("%s Fisher results for %s k=%d:\n%s", method_label, metric_name, k, res)
    return results


def run_fisher_loop_precomputed(
    top_clusters: dict,
    fisher_fn,
    base_kwargs: dict,
    dist_matrix,
    df_ids: pd.DataFrame,
    logger: logging.Logger,
    method_label: str,
    dist_arg_name: str = "dist_matrix_filled",  # name expected by fisher_fn
    ids_arg_name: str = "df_ids"
) -> dict:
    """
    Same idea as run_fisher_loop, but injects a precomputed distance matrix
    and an id DataFrame each time.
    """
    results = {}
    for metric_name, ks in top_clusters.items():
        if not ks:
            logger.info("No k selected for %s (%s); skipping Fisher.", metric_name, method_label)
            continue
        for k in ks:
            kwargs = base_kwargs.copy()
            kwargs["k"] = k
            kwargs[dist_arg_name] = dist_matrix
            kwargs[ids_arg_name] = df_ids
            res = fisher_fn(**kwargs)
            key = f"{metric_name}_k={k}"
            results[key] = res
            logger.info("%s Fisher results for %s k=%d:\n%s", method_label, metric_name, k, res)
    return results

def run_fisher_loop_spectral(
    top_clusters: dict,
    feature_matrix: np.ndarray,
    ids_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    outcome_col: str,
    logger: logging.Logger,
    method_label: str = "Spectral",
    affinity: str = "nearest_neighbors",
    random_state: int = 42,
) -> dict:
    """
    For each k in top_clusters, fit SpectralClustering, then run Fisher.
    Returns { "<metric>_k=<k>": result_df }
    """
    results = {}
    for metric_name, ks in top_clusters.items():
        if not ks:
            logger.info("No k selected for %s (%s); skipping Fisher.", metric_name, method_label)
            continue
        for k in ks:
            spec = SpectralClustering(n_clusters=k, affinity=affinity, random_state=random_state)
            labels = spec.fit_predict(feature_matrix)
            res = run_fisher_test_spectral(labels, ids_df, outcome_df, outcome_col)
            key = f"{metric_name}_k={k}"
            results[key] = res
            logger.info("%s Fisher results for %s k=%d:\n%s", method_label, metric_name, k, res)
    return results
    
def run_dbscan_block_euclidean(
    df_features: pd.DataFrame,
    id_vars: list[str],
    k_for_elbow: int,
    eps: float,
    min_samples: int,
    outcome_df: pd.DataFrame,
    outcome_col: str,
    logger: logging.Logger,
    prefix: str = "DBSCAN-Euclidean"
):
    """
    1) Build X
    2) k-distance elbow + save plot
    3) run_dbscan_euclidean
    4) log metrics & fisher
    Returns (metrics_df, labels, fisher_dict)
    """
    feature_cols = [c for c in df_features.columns if c not in id_vars]
    X = df_features[feature_cols].values

    kdist = kdist_sorted(X, k=k_for_elbow)
    utils.plot_and_save_kdist(kdist, filename=f"{prefix.lower()}_kdist.png",
                              k_label=f"{k_for_elbow}-NN distance")

    metrics_df, labels, fisher_dict = run_dbscan_euclidean(
        df_features=df_features,
        id_vars=id_vars,
        eps=eps,
        min_samples=min_samples,
        outcome_df=outcome_df,
        outcome_col=outcome_col
    )

    logger.info("%s metrics:\n%s", prefix, metrics_df)
    if "results" in fisher_dict:
        logger.info("%s Fisher results:\n%s", prefix, fisher_dict["results"])

    return metrics_df, labels, fisher_dict

def run_dbscan_block_precomputed(
    dist_matrix: np.ndarray,
    df_ids: pd.DataFrame,
    feature_space_X: np.ndarray,
    k_for_elbow: int,
    eps: float,
    min_samples: int,
    outcome_df: pd.DataFrame,
    outcome_col: str,
    logger: logging.Logger,
    prefix: str = "DBSCAN-Precomputed",
    dist_label: str = "DTW"
):
    """
    1) k-distance elbow on precomputed matrix
    2) run_dbscan_precomputed
    3) log metrics & fisher
    Returns (metrics_df, labels, fisher_dict)
    """
    kdist_pre = kdist_sorted_precomputed(dist_matrix, k=k_for_elbow)
    utils.plot_and_save_kdist_precomputed(
        kdist_pre,
        filename=f"{prefix.lower()}_kdist.png",
        k_label=f"{k_for_elbow}-NN dist ({dist_label})"
    )

    metrics_df, labels, fisher_dict = run_dbscan_precomputed(
        dist_matrix=dist_matrix,
        df_ids=df_ids,
        eps=eps,
        min_samples=min_samples,
        feature_space_X=feature_space_X,
        outcome_df=outcome_df,
        outcome_col=outcome_col
    )

    logger.info("%s metrics:\n%s", prefix, metrics_df)
    if "results" in fisher_dict:
        logger.info("%s Fisher results:\n%s", prefix, fisher_dict["results"])

    return metrics_df, labels, fisher_dict

def spectral_metrics_block(
    feature_matrix: np.ndarray,
    k_values: range,
    affinity: str,
    plot_filename: str,
    sil_threshold: float,
    logger: logging.Logger
):
    """Run spectral metrics, plot curves, pick top clusters."""
    metrics_df = run_spectral_metrics(
        feature_matrix=feature_matrix,
        k_values=k_values,
        affinity=affinity,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(metrics_df, filename=plot_filename)
    top_clust = get_top_clusters_with_threshold(metrics_df, sil_threshold, logger=logger)
    return metrics_df, top_clust


def should_run(name: str, run_all: bool, flags: dict[str, bool]) -> bool:
    return run_all or flags.get(name, False)
