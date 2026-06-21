from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def reduce_dimensions(X_train, X_test, n_components=10):
    pca = PCA(n_components)
    x_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return (x_train_pca, X_test_pca, pca)

def run_kmeans(x_pca, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_run = kmeans.fit(x_pca)
    return(kmeans_run)

def evaluate_clusters(x_pca, kmeans_run):
    inertia = kmeans_run.inertia_
    sample_silhouette_score = silhouette_score(x_pca, kmeans_run.labels_)
    return(inertia, sample_silhouette_score)