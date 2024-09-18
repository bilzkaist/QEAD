from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score

def classical_methods(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': GaussianNB(),
        'SVM (Radial)': SVC(kernel='rbf', probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
        'Elliptic Envelope': EllipticEnvelope(),
        'Isolation Forest': IsolationForest()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'accuracy': accuracy}
    
    return results
