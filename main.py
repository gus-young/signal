from analysis.preprocessor import get_feature_types, encode_categoricals, scale_features, build_feature_matrix
import numpy as np 
import matplotlib.pyplot as plt
from analysis.cluster import reduce_dimensions, run_kmeans, evaluate_clusters

# Train Data
X_train = np.load('output/X_train.npy')
y_train = np.load('output/y_train.npy', allow_pickle=True)

# Test Data
X_test = np.load('output/X_test.npy')
y_test = np.load('output/y_test.npy', allow_pickle=True)

x_pca = reduce_dimensions(X_train, X_test)
print(run_kmeans(x_pca[1]))

x = []
y = []

for num in range (2,11):
    kmeans_run = run_kmeans(x_pca[1], k=num)
    clusters = evaluate_clusters (x_pca[1], kmeans_run)
    x.append(num)
    y.append(clusters[0])

print (x)
print (y)

plt.plot(x, y)
plt.xlabel = "k value"
plt.ylabel = "Inertia Value"
plt.title = "k vs Inertia"
plt.savefig("plots/elbow.png")