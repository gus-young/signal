from sklearn.decomposition import PCA 

def reduce_dimensions(X_train, X_test, n_components=10):
    pca = PCA(n_components)
    x_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return (x_train_pca, X_test_pca, pca)