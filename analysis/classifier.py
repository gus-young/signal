from sklearn.ensemble import RandomForestClassifier

def train_classifier(X_train, y_train):
    rf_class = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    fit_model = rf_class.fit(X_train, y_train)
    return fit_model

