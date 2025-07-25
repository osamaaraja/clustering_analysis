from hyperparameters import *
import utils
from imports import *
from clustering import *

dtw_ts = None

utils.configure_logging()
logger = logging.getLogger('MAIN')

start = time.time()

data_path_cohort = "data/cohort1_target.csv"
data_path_labevent = "data/labevents.csv"

cohort_data = utils.loading_data(data_path_cohort,  'dischtime')
labevents_data = utils.loading_data(data_path_labevent, 'charttime', 5000)

elapsed_time = time.time() - start
logger.info("Execution time to load the data: %d minutes and %d seconds", int(elapsed_time//60), int(elapsed_time%60))

merged = utils.merge_data(labevents_data, cohort_data)
logger.info("Data merged successfully")

result = utils.labs_within_n_days_of_discharge(merged, days)
logger.info("Filtered lab results are successfully taken within the previous %d days.", days)

# ---------- single percentage pipeline ----------
filtered_common_labs_percent, unique_labs, total_admissions = utils.filtering_labs_by_percentage(result, set_percentage)
logger.info("Total unique item_ids (lab tests) are %d and Total admissions (patients) are %d.", unique_labs, total_admissions)

pivot_percent = utils.descritization(filtered_common_labs_percent, days)

feature_matrix_percent, feature_cols_percent, id_vars = utils.get_feature_matrix(pivot_percent)
normalized_long_df_percent = utils.create_long_format(pivot_percent, feature_cols_percent, id_vars)
imputed_df_percent, wide_df_percent = utils.get_imputed_df(normalized_long_df_percent, id_vars)
imputed_preprocessed = utils.knn_impute_and_sort_features(wide_df_percent, id_vars)

# label helper (optional)
PCT_LABEL = f"{int(set_percentage*100)}pct"

# ------------------- KMEANS --------------------
if utils.should_run("KMEANS", RUN_ALL, RUN_FLAGS):
    metrics_df_km = run_kmeans_metrics(imputed_preprocessed, id_vars, K_RANGE, logger=logger)
    utils.plot_and_save_kmeans_metrics(metrics_df_km, filename=f"kmeans_metrics_{PCT_LABEL}.png")

    top_clust_km = get_top_clusters_with_threshold(metrics_df_km, SIL_THR, logger=logger)

    all_fisher_km = utils.run_fisher_loop(
        top_clusters=top_clust_km,
        fisher_fn=run_fisher_test,
        base_kwargs=dict(
            df_features=imputed_preprocessed,
            id_vars=id_vars,
            outcome_df=cohort_data,
            outcome_col='target',
            random_state=42,
            n_init=10
        ),
        logger=logger,
        method_label=f"KMeans-{PCT_LABEL}"
    )
else:
    metrics_df_km = top_clust_km = all_fisher_km = None


# ------------------- AGGLO EUCLIDEAN (feature space) --------------------
if utils.should_run("AGG_EUCLID", RUN_ALL, RUN_FLAGS):
    metrics_df_agg_eu = run_agglomerative_metrics_euclidean(imputed_preprocessed, id_vars, K_RANGE, logger=logger)
    utils.plot_and_save_kmeans_metrics(metrics_df_agg_eu, filename=f"agg_metrics_{PCT_LABEL}_eu.png")

    top_clust_agg_eu = get_top_clusters_with_threshold(metrics_df_agg_eu, SIL_THR, logger=logger)

    all_fisher_agg_eu = utils.run_fisher_loop(
        top_clusters=top_clust_agg_eu,
        fisher_fn=run_fisher_test_agglomerative_euclidean,
        base_kwargs=dict(
            df_features=imputed_preprocessed,
            id_vars=id_vars,
            outcome_df=cohort_data,
            outcome_col="target",
            linkage="ward",
            metric="euclidean"
        ),
        logger=logger,
        method_label=f"Agglo-Euclidean-{PCT_LABEL}"
    )
else:
    metrics_df_agg_eu = top_clust_agg_eu = all_fisher_agg_eu = None


# ------------------- AGGLO MANHATTAN (precomputed) --------------------
if utils.should_run("AGG_MANHATTAN", RUN_ALL, RUN_FLAGS):
    metrics_df_agg_man, dist_man = run_agglomerative_metrics_manhattan(
        df_raw=wide_df_percent,
        id_vars=id_vars,
        k_values=K_RANGE,
        df_for_space=imputed_preprocessed,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(metrics_df_agg_man, filename=f"agg_metrics_{PCT_LABEL}_manhattan.png")

    top_clust_agg_man = get_top_clusters_with_threshold(metrics_df_agg_man, SIL_THR, logger=logger)

    all_fisher_agg_man = utils.run_fisher_loop_precomputed(
        top_clusters=top_clust_agg_man,
        fisher_fn=run_fisher_test_agglomerative_manhattan,
        base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
        dist_matrix=dist_man,
        df_ids=wide_df_percent[id_vars],
        logger=logger,
        method_label=f"Agglo-Manhattan-{PCT_LABEL}"
    )
else:
    metrics_df_agg_man = dist_man = top_clust_agg_man = all_fisher_agg_man = None


# ------------------- AGGLO MAHALANOBIS --------------------
if utils.should_run("AGG_MAHALANOBIS", RUN_ALL, RUN_FLAGS):
    metrics_df_agg_mah, dist_mah = run_agglomerative_metrics_mahalanobis(
        df_raw=wide_df_percent,
        id_vars=id_vars,
        k_values=K_RANGE,
        df_for_space=imputed_preprocessed,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(metrics_df_agg_mah, filename=f"agg_metrics_{PCT_LABEL}_mahalanobis.png")

    top_clust_agg_mah = get_top_clusters_with_threshold(metrics_df_agg_mah, SIL_THR, logger=logger)

    all_fisher_agg_mah = utils.run_fisher_loop_precomputed(
        top_clusters=top_clust_agg_mah,
        fisher_fn=run_fisher_test_agglomerative_mahalanobis,
        base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
        dist_matrix=dist_mah,
        df_ids=wide_df_percent[id_vars],
        logger=logger,
        method_label=f"Agglo-Mahalanobis-{PCT_LABEL}"
    )
else:
    metrics_df_agg_mah = dist_mah = top_clust_agg_mah = all_fisher_agg_mah = None


# ------------------- AGGLO COSINE --------------------
if utils.should_run("AGG_COSINE", RUN_ALL, RUN_FLAGS):
    metrics_df_agg_cos, dist_cos = run_agglomerative_metrics_cosine(
        df_raw=wide_df_percent,
        id_vars=id_vars,
        k_values=K_RANGE,
        df_for_space=imputed_preprocessed,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(metrics_df_agg_cos, filename=f"agg_metrics_{PCT_LABEL}_cosine.png")

    top_clust_agg_cos = get_top_clusters_with_threshold(metrics_df_agg_cos, SIL_THR, logger=logger)

    all_fisher_agg_cos = utils.run_fisher_loop_precomputed(
        top_clusters=top_clust_agg_cos,
        fisher_fn=run_fisher_test_agglomerative_cosine,
        base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
        dist_matrix=dist_cos,
        df_ids=wide_df_percent[id_vars],
        logger=logger,
        method_label=f"Agglo-Cosine-{PCT_LABEL}"
    )
else:
    metrics_df_agg_cos = dist_cos = top_clust_agg_cos = all_fisher_agg_cos = None


# ------------------- AGGLO EUCLIDEAN (precomputed) --------------------
if utils.should_run("AGG_EU_PRE", RUN_ALL, RUN_FLAGS):
    metrics_df_agg_eu_pre, dist_eu_pre = run_agglomerative_metrics_euclidean_precomputed(
        df_raw=wide_df_percent,
        id_vars=id_vars,
        k_values=K_RANGE,
        df_for_space=imputed_preprocessed,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(metrics_df_agg_eu_pre, filename=f"agg_metrics_{PCT_LABEL}_euclid_precomputed.png")

    top_clust_agg_eu_pre = get_top_clusters_with_threshold(metrics_df_agg_eu_pre, SIL_THR, logger=logger)

    all_fisher_agg_eu_pre = utils.run_fisher_loop_precomputed(
        top_clusters=top_clust_agg_eu_pre,
        fisher_fn=run_fisher_test_agglomerative_euclidean_precomputed,
        base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
        dist_matrix=dist_eu_pre,
        df_ids=wide_df_percent[id_vars],
        logger=logger,
        method_label=f"Agglo-Euclid-precomputed-{PCT_LABEL}"
    )
else:
    metrics_df_agg_eu_pre = dist_eu_pre = top_clust_agg_eu_pre = all_fisher_agg_eu_pre = None


# ------------------- AGGLO DTW (dtaidistance) --------------------
if utils.should_run("AGG_DTW_FAST", RUN_ALL, RUN_FLAGS):
    metrics_df_agg_dtw, dtw_mat = run_agglomerative_metrics_dtw(
        df_features=imputed_preprocessed,
        id_vars=id_vars,
        n_timesteps=days,
        k_values=K_RANGE,
        df_for_space=imputed_preprocessed,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(metrics_df_agg_dtw, filename=f"agg_metrics_{PCT_LABEL}_dtw.png")

    top_clust_agg_dtw = get_top_clusters_with_threshold(metrics_df_agg_dtw, SIL_THR, logger=logger)

    all_fisher_agg_dtw = utils.run_fisher_loop_precomputed(
        top_clusters=top_clust_agg_dtw,
        fisher_fn=run_fisher_test_agglomerative_dtw,
        base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
        dist_matrix=dtw_mat,
        df_ids=imputed_preprocessed[id_vars],
        logger=logger,
        method_label=f"Agglo-DTW-{PCT_LABEL}",
        dist_arg_name="dtw_matrix",
        ids_arg_name="df_ids"
    )
else:
    metrics_df_agg_dtw = dtw_mat = top_clust_agg_dtw = all_fisher_agg_dtw = None

# ------------------- BINARY (Hamming/Jaccard/Dice) --------------------
if utils.should_run("BINARY", RUN_ALL, RUN_FLAGS):
    binary_dataset = build_binary_dataset(filtered_common_labs_percent, id_col="hadm_id", item_col="itemid")
    bin_feats_np, bin_feats_bool_np, dist_mats_bin = compute_binary_distance_matrices(binary_dataset, id_col="hadm_id")

    metrics_dfs_bin = run_agglomerative_metrics_binary(
        dist_mats=dist_mats_bin,
        feature_space_np=bin_feats_np,
        k_values=K_RANGE,
        logger=logger
    )
    for name, df in metrics_dfs_bin.items():
        utils.plot_and_save_kmeans_metrics(df, filename=f"agg_metrics_{PCT_LABEL}_binary_{name.lower()}.png")

    all_fisher_agg_bin = {}
    for name, df in metrics_dfs_bin.items():
        top_clust_bin = get_top_clusters_with_threshold(df, SIL_THR, logger=logger)
        logger.info("Top clusters for %s: %s", name, top_clust_bin)
        res_dict = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_bin,
            fisher_fn=run_fisher_test_agglomerative_binary,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_mats_bin[name],
            df_ids=binary_dataset[["hadm_id"]],
            logger=logger,
            method_label=f"Binary-{name}-{PCT_LABEL}"
        )
        all_fisher_agg_bin.update({f"{name}_{k}": v for k, v in res_dict.items()})
else:
    binary_dataset = dist_mats_bin = metrics_dfs_bin = all_fisher_agg_bin = None


# ------------------- DBSCAN (Euclidean) --------------------
if utils.should_run("DBSCAN_EU", RUN_ALL, RUN_FLAGS):
    dbscan_metrics_eu, dbscan_labels_eu, fisher_eu_dbscan = utils.run_dbscan_block_euclidean(
        df_features=imputed_preprocessed,
        id_vars=id_vars,
        k_for_elbow=5,
        eps=20,
        min_samples=5,
        outcome_df=cohort_data,
        outcome_col="target",
        logger=logger,
        prefix=f"DBSCAN-Euclidean-{PCT_LABEL}"
    )
else:
    dbscan_metrics_eu = dbscan_labels_eu = fisher_eu_dbscan = None


# ------------------- DBSCAN (DTW) --------------------
if utils.should_run("DBSCAN_DTW", RUN_ALL, RUN_FLAGS):
    # ensure D_dtw exists (fallback to dtw_ts if only that was computed)
    if 'dtw_mat' in locals():
        D_dtw = dtw_mat
    elif 'dtw_ts' in locals():
        D_dtw = dtw_ts
    else:
        raise RuntimeError("DTW matrix not available. Enable AGG_DTW_FAST or AGG_DTW_TS first.")

    df_ids = imputed_preprocessed[id_vars]
    X_space = imputed_preprocessed.drop(columns=id_vars).values

    dbscan_metrics_dtw, dbscan_labels_dtw, fisher_dtw_dbscan = utils.run_dbscan_block_precomputed(
        dist_matrix=D_dtw,
        df_ids=df_ids,
        feature_space_X=X_space,
        k_for_elbow=5,
        eps=15,
        min_samples=10,
        outcome_df=cohort_data,
        outcome_col="target",
        logger=logger,
        prefix=f"DBSCAN-DTW-{PCT_LABEL}",
        dist_label="DTW"
    )
else:
    dbscan_metrics_dtw = dbscan_labels_dtw = fisher_dtw_dbscan = None


# ------------------- SPECTRAL --------------------
if utils.should_run("SPECTRAL", RUN_ALL, RUN_FLAGS):
    hmm_feats, ids_df = build_hmm_state_features(
        df_features=imputed_preprocessed,
        id_vars=id_vars,
        n_timesteps=days,
        n_hmm_states=3,
        logger=logger
    )

    metrics_df_spec, top_clust_spec = utils.spectral_metrics_block(
        feature_matrix=hmm_feats,
        k_values=K_RANGE,
        affinity="nearest_neighbors",
        plot_filename=f"spectral_metrics_{PCT_LABEL}.png",
        sil_threshold=SIL_THR,
        logger=logger
    )

    all_fisher_spec = utils.run_fisher_loop_spectral(
        top_clusters=top_clust_spec,
        feature_matrix=hmm_feats,
        ids_df=ids_df,
        outcome_df=cohort_data,
        outcome_col="target",
        logger=logger,
        method_label=f"Spectral-{PCT_LABEL}"
    )
else:
    hmm_feats = ids_df = metrics_df_spec = top_clust_spec = all_fisher_spec = None

#---------------------------------------------------------------------------------------#
# ------------------- K‑MEDOIDS EUCLIDEAN --------------------
if utils.should_run("KMEDOIDS_EUCLID", RUN_ALL, RUN_FLAGS):
    metrics_kmed_eu = run_kmedoids_metrics(
        imputed_preprocessed, id_vars,
        k_values=K_RANGE,
        metric="euclidean", precomputed=False,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(
        metrics_kmed_eu,
        filename=f"kmedoids_metrics_{PCT_LABEL}_euclid.png"
    )
    top_kmed_eu = get_top_clusters_with_threshold(metrics_kmed_eu, SIL_THR, logger=logger)
    all_fisher_kmed_eu = utils.run_fisher_loop(
        top_clusters=top_kmed_eu,
        fisher_fn=run_fisher_test_kmedoids,
        base_kwargs=dict(
            df_features=imputed_preprocessed,
            id_vars=id_vars,
            outcome_df=cohort_data,
            outcome_col="target",
            metric="euclidean",
            precomputed=False,
            random_state=42
        ),
        logger=logger,
        method_label=f"KMedoids-EU-{PCT_LABEL}"
    )
else:
    metrics_kmed_eu = top_kmed_eu = all_fisher_kmed_eu = None

# ------------------- K‑MEDOIDS MANHATTAN (precomputed) --------------------
if utils.should_run("KMEDOIDS_MANHATTAN", RUN_ALL, RUN_FLAGS):
    # first compute your manhattan dist matrix exactly as for Agglo
    _, man_dist = run_agglomerative_metrics_manhattan(
        df_raw=wide_df_percent,
        id_vars=id_vars,
        k_values=K_RANGE,
        df_for_space=None,    # not used here
        logger=logger
    )
    metrics_kmed_man = run_kmedoids_metrics(
        imputed_preprocessed, id_vars,
        k_values=K_RANGE,
        metric="precomputed", precomputed=True,
        dist_matrix=man_dist,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(metrics_kmed_man, filename=f"kmedoids_metrics_{PCT_LABEL}_manhattan.png")
    top_kmed_man = get_top_clusters_with_threshold(metrics_kmed_man, SIL_THR, logger=logger)
    all_fisher_kmed_man = utils.run_fisher_loop_precomputed(
        top_clusters=top_kmed_man,
        fisher_fn=run_fisher_test_kmedoids,
        base_kwargs=dict(
            outcome_df=cohort_data,
            outcome_col="target",
            metric="precomputed",
            precomputed=True
        ),
        dist_matrix=man_dist,
        df_ids=wide_df_percent[id_vars],
        logger=logger,
        method_label=f"KMedoids-MAN-{PCT_LABEL}"
    )
else:
    metrics_kmed_man = dist_man = top_kmed_man = all_fisher_kmed_man = None

# ------------------- K‑MEDOIDS MAHALANOBIS (precomputed) --------------------
if utils.should_run("KMEDOIDS_MAHALANOBIS", RUN_ALL, RUN_FLAGS):
    # uses dist_mah from your Agglo‑Mahalanobis block
    metrics_kmed_mah = run_kmedoids_metrics(
        imputed_preprocessed, id_vars,
        k_values=K_RANGE,
        metric="precomputed", precomputed=True,
        dist_matrix=dist_mah,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(
        metrics_kmed_mah,
        filename=f"kmedoids_metrics_{PCT_LABEL}_mahalanobis.png"
    )
    top_kmed_mah = get_top_clusters_with_threshold(
        metrics_kmed_mah, SIL_THR, logger=logger
    )
    all_fisher_kmed_mah = utils.run_fisher_loop_precomputed(
        top_clusters=top_kmed_mah,
        fisher_fn=run_fisher_test_kmedoids,
        base_kwargs=dict(
            outcome_df=cohort_data,
            outcome_col="target",
            metric="precomputed",
            precomputed=True
        ),
        dist_matrix=dist_mah,
        df_ids=wide_df_percent[id_vars],
        logger=logger,
        method_label=f"KMedoids‑MAH-{PCT_LABEL}"
    )
else:
    metrics_kmed_mah = top_kmed_mah = all_fisher_kmed_mah = None


# ------------------- K‑MEDOIDS COSINE (precomputed) --------------------
if utils.should_run("KMEDOIDS_COSINE", RUN_ALL, RUN_FLAGS):
    # uses dist_cos from your Agglo‑Cosine block
    metrics_kmed_cos = run_kmedoids_metrics(
        imputed_preprocessed, id_vars,
        k_values=K_RANGE,
        metric="precomputed", precomputed=True,
        dist_matrix=dist_cos,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(
        metrics_kmed_cos,
        filename=f"kmedoids_metrics_{PCT_LABEL}_cosine.png"
    )
    top_kmed_cos = get_top_clusters_with_threshold(
        metrics_kmed_cos, SIL_THR, logger=logger
    )
    all_fisher_kmed_cos = utils.run_fisher_loop_precomputed(
        top_clusters=top_kmed_cos,
        fisher_fn=run_fisher_test_kmedoids,
        base_kwargs=dict(
            outcome_df=cohort_data,
            outcome_col="target",
            metric="precomputed",
            precomputed=True
        ),
        dist_matrix=dist_cos,
        df_ids=wide_df_percent[id_vars],
        logger=logger,
        method_label=f"KMedoids‑COS-{PCT_LABEL}"
    )
else:
    metrics_kmed_cos = top_kmed_cos = all_fisher_kmed_cos = None


# ------------------- K‑MEDOIDS DTW (fast, dtaidistance) --------------------
if utils.should_run("KMEDOIDS_DTW_FAST", RUN_ALL, RUN_FLAGS):
    # uses dtw_mat from your DTW‑fast block
    metrics_kmed_dtw, _ = run_kmedoids_metrics(
        imputed_preprocessed, id_vars,
        k_values=K_RANGE,
        metric="precomputed", precomputed=True,
        dist_matrix=dtw_mat,
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(
        metrics_kmed_dtw,
        filename=f"kmedoids_metrics_{PCT_LABEL}_dtw.png"
    )
    top_kmed_dtw = get_top_clusters_with_threshold(
        metrics_kmed_dtw, SIL_THR, logger=logger
    )
    all_fisher_kmed_dtw = utils.run_fisher_loop_precomputed(
        top_clusters=top_kmed_dtw,
        fisher_fn=run_fisher_test_kmedoids,
        base_kwargs=dict(
            outcome_df=cohort_data,
            outcome_col="target",
            metric="precomputed",
            precomputed=True
        ),
        dist_matrix=dtw_mat,
        df_ids=imputed_preprocessed[id_vars],
        logger=logger,
        method_label=f"KMedoids‑DTW‑FAST-{PCT_LABEL}",
        dist_arg_name="dtw_matrix",
        ids_arg_name="df_ids"
    )
else:
    metrics_kmed_dtw = top_kmed_dtw = all_fisher_kmed_dtw = None


# ------------------- K‑MEDOIDS DTW (tslearn) --------------------
if utils.should_run("KMEDOIDS_DTW_TS", RUN_ALL, RUN_FLAGS):
    # ensure dtw_ts has been computed
    if 'dtw_ts' not in locals():
        raise RuntimeError(
            "K‑Medoids DTW‑TS requires the tslearn‑DTW matrix. "
            "Enable and run AGG_DTW_TS first."
        )

    metrics_kmed_dtw_ts, _ = run_kmedoids_metrics(
        imputed_preprocessed, id_vars,
        k_values=K_RANGE,
        metric="precomputed", precomputed=True,
        dist_matrix=dtw_ts,     # now guaranteed to exist
        logger=logger
    )
    utils.plot_and_save_kmeans_metrics(
        metrics_kmed_dtw_ts,
        filename=f"kmedoids_metrics_{PCT_LABEL}_dtw_tslearn.png"
    )
    top_kmed_dtw_ts = get_top_clusters_with_threshold(
        metrics_kmed_dtw_ts, SIL_THR, logger=logger
    )
    all_fisher_kmed_dtw_ts = utils.run_fisher_loop_precomputed(
        top_clusters=top_kmed_dtw_ts,
        fisher_fn=run_fisher_test_kmedoids,
        base_kwargs=dict(
            outcome_df=cohort_data,
            outcome_col="target",
            metric="precomputed",
            precomputed=True
        ),
        dist_matrix=dtw_ts,
        df_ids=imputed_preprocessed[id_vars],
        logger=logger,
        method_label=f"KMedoids‑DTW‑TS-{PCT_LABEL}",
        dist_arg_name="dtw_matrix",
        ids_arg_name="df_ids"
    )
else:
    metrics_kmed_dtw_ts = top_kmed_dtw_ts = all_fisher_kmed_dtw_ts = None
