# Contains all the imports

import numpy as np
import pandas as pd
from datetime import timedelta
import time
import os
os.environ["OMP_NUM_THREADS"] = "20" # to remove the warning
import logging
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.metrics.pairwise import cosine_distances
from tslearn.metrics import cdist_dtw
from sklearn.cluster import AgglomerativeClustering
from numpy.linalg import inv, LinAlgError
from scipy.linalg import pinv
from dtaidistance import dtw
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from hmmlearn import hmm
from sklearn.cluster import SpectralClustering
from dataclasses import dataclass 
from typing import Iterable, List, Dict

#from sklearn_extra.cluster import KMedoids




