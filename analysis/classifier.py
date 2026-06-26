from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
def train_classifier(X_train, y_train):
    rf_class = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    fit_model = rf_class.fit(X_train, y_train)
    return fit_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_pred