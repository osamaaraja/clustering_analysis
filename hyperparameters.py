days = 7

K_RANGE = range(2, 16)
SIL_THR = 0.25


set_percentage = 1/100

RUN_ALL = False

RUN_FLAGS = {
    "KMEANS": True,
    "AGG_EUCLID": True,
    "AGG_MANHATTAN": False,
    "AGG_MAHALANOBIS": False,
    "AGG_COSINE": False,
    "AGG_EU_PRE": False,
    "AGG_DTW_FAST": False,
    "AGG_DTW_TS": False,
    "BINARY": False,
    "DBSCAN_EU": False,
    "DBSCAN_DTW": False,
    "SPECTRAL": False,
    "KMEDOIDS_EUCLID": True,
    "KMEDOIDS_MANHATTAN": False,
    "KMEDOIDS_MAHALANOBIS": False,
    "KMEDOIDS_COSINE": False,
    "KMEDOIDS_DTW_FAST": False,
    "KMEDOIDS_DTW_TS": False,
}
