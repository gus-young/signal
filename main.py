from analysis.preprocessor import get_feature_types, encode_categoricals, scale_features, build_feature_matrix
from sklearn.decomposition import PCA 
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from analysis.cluster import reduce_dimensions, run_kmeans, evaluate_clusters
from datetime import datetime

# Train Data
X_train = np.load('output/X_train.npy')
y_train = np.load('output/y_train.npy', allow_pickle=True)

# Test Data
X_test = np.load('output/X_test.npy')
y_test = np.load('output/y_test.npy', allow_pickle=True)

# Reduce Dimensions of data 
x_pca = reduce_dimensions(X_train, X_test, 30)

def evaluate_cluster_plot(x_pca, n=5):
    x = []
    y = []

    for num in tqdm(range(2, n+1)):
        kmeans_run = run_kmeans(x_pca[0], k=num)
        clusters = evaluate_clusters (x_pca[0], kmeans_run)
        x.append(num)
        y.append(clusters[0])

    plt.plot(x, y)
    plt.xlabel("k value")
    plt.ylabel("Inertia Value")
    plt.title ("k vs Inertia")
    plt.savefig(f"plots/elbow_k{n}.png")

    # Calculate second derivative for approximate k value 
    deriv_2 = np.gradient(np.gradient(y, x),x)
    elbow = x[np.argmax(deriv_2)]
    
    # Generate Report
    print( "-- Run Report ---") 
    print ("Timestamp: ", datetime.now())
    print(f"Second Deriv: {elbow}")
    print ("Silhouette Score:", clusters[1])
    

def plot_2d(x_pca):
    kmeans_run = run_kmeans(x_pca[0], 5)
    x_pca2d = PCA(n_components=2)
    X_2d = x_pca2d.fit_transform(X_train)
    x = X_2d[:, 0]
    y = X_2d[:, 1]
    plt.scatter(x, y, c=kmeans_run.labels_, cmap="tab10", alpha=0.3, s=1)
    plt.savefig("plots/clusters_kmeans.png")
                              
evaluate_cluster_plot(x_pca, 10)
#plot_2d(x_pca)
