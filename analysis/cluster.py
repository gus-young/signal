from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

def reduce_dimensions(X_train, X_test, n_components=20):
    pca = PCA(n_components)
    x_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    variance_ratios = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratios)
    #print(variance_ratios, cumulative_variance)
    return (x_train_pca, X_test_pca, pca)

def run_kmeans(x_pca, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_run = kmeans.fit(x_pca)
    return(kmeans_run)

def evaluate_clusters(x_pca, kmeans_run):
    inertia = kmeans_run.inertia_
    sample_silhouette_score = silhouette_score(x_pca, kmeans_run.labels_)
    return(inertia, sample_silhouette_score)

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

def plot_2d(x_pca, X_train):
    kmeans_run = run_kmeans(x_pca[0], 5)
    x_pca2d = PCA(n_components=2)
    X_2d = x_pca2d.fit_transform(X_train)
    x = X_2d[:, 0]
    y = X_2d[:, 1]
    plt.scatter(x, y, c=kmeans_run.labels_, cmap="tab10", alpha=0.3, s=1)
    plt.savefig("plots/clusters_kmeans.png")