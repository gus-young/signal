from analysis.preprocessor import get_feature_types, encode_categoricals, scale_features, build_feature_matrix
from sklearn.decomposition import PCA 
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from analysis.cluster import reduce_dimensions, run_kmeans, evaluate_clusters, evaluate_cluster_plot, plot_2d
from datetime import datetime

# Train Data
X_train = np.load('output/X_train.npy')
y_train = np.load('output/y_train.npy', allow_pickle=True)

# Test Data
X_test = np.load('output/X_test.npy')
y_test = np.load('output/y_test.npy', allow_pickle=True)

# Reduce Dimensions of data 
x_pca = reduce_dimensions(X_train, X_test, 20)
                              
#evaluate_cluster_plot(x_pca, 5)
plot_2d(x_pca, X_train, y_train)

