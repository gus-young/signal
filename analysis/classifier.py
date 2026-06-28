from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_classifier(X_train, y_train):
    rf_class = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    fit_model = rf_class.fit(X_train, y_train)
    return fit_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_pred

def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plot_size = len(labels)

    for i in range(plot_size):
        for j in range (plot_size):
            plt.text(j, i, cm[i,j],ha='center',va='center')
    plt.savefig("plots/confusion_matrix.png")