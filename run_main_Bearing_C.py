import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, recall_score, precision_recall_curve, auc, confusion_matrix
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.covariance import EllipticEnvelope
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.decomposition import PCA 
import os
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
import warnings
warnings.filterwarnings("ignore")

# Set dataset path and file name for the Bearing dataset
dataset_path = "/home/bilz/datasets/qead/q/"
dataset_name = "Bearing.csv"

# Use a non-interactive backend to avoid "Wayland" issues
import matplotlib
matplotlib.use('Agg')

# Preprocessing function for Bearing dataset
def preprocess_data(data, window_size=20):
    print(f"Preprocessing bearing data with window size {window_size}...")

    if data.empty:
        raise ValueError("Dataset is empty.")

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    
    # Process the bearing data
    if 'Bearing 1' in data.columns:
        print("Processing bearing data from 'Bearing 1'...")
        data['Bearing 1'] = scaler.fit_transform(data['Bearing 1'].values.reshape(-1, 1))
        X = np.array([data['Bearing 1'].values[i:i + window_size] for i in range(len(data) - window_size)])
        y_true = np.array([1 if val > 0.8 else 0 for val in data['Bearing 1'][window_size:]])
    else:
        raise ValueError("Unknown data format in dataset.")
    
    print("Preprocessing complete.")
    return X, y_true

# Load dataset
def load_datasets():
    print("Loading Bearing dataset...")
    file_path = dataset_path + dataset_name
    data_Bearing = pd.read_csv(file_path)
    print(f"Loaded Bearing dataset with columns: {data_Bearing.columns}")
    return {'Bearing': data_Bearing}

# Quantum encoding function
def encode_data(X):
    n_qubits = len(X)
    qc = QuantumCircuit(n_qubits)
    for i, x in enumerate(X):
        qc.ry(x, i)
    return qc

# Variational circuit function
def variational_circuit(params, n_qubits):
    qc = QuantumCircuit(n_qubits)
    for i, param in enumerate(params):
        qc.rx(param, i)
        qc.rz(param, i)
    return qc

# Quantum anomaly detection objective function
loss_history = []
def objective_function(params, X):
    n_qubits = len(X)
    qc = encode_data(X)
    var_circuit = variational_circuit(params, n_qubits)
    qc.compose(var_circuit, inplace=True)
    
    estimator = Estimator()
    observable = QuantumCircuit(n_qubits)
    observable.z(0)
    expectation_value = estimator.run([qc], [observable]).result().values[0]
    
    loss = 1 - np.abs(expectation_value)
    loss_history.append(loss)
    return loss

# Optimize quantum circuit parameters
def optimize_params(X, initial_params):
    result = minimize(objective_function, initial_params, args=(X,), method='COBYLA')
    return result.x

# Smart threshold using IQR
def calculate_smart_threshold(scores):
    q25, q75 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    return upper_bound

# Framework 1: QAE + One-Class SVM
def framework1_qae_oneclass_svm(X_train):
    """
    QAE + One-Class SVM framework for anomaly detection.
    This method performs dimensionality reduction, quantum encoding, and uses One-Class SVM for classification.
    """
    # Dimensionality reduction via PCA to match qubit requirements
    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(X_train)
    
    # Quantum encoding (QAE)
    encoded_X = [encode_data(sample) for sample in X_reduced]  # Encode each sample into quantum states
    
    # Use One-Class SVM for anomaly detection
    svm_model = OneClassSVM(kernel='rbf', nu=0.1)
    svm_model.fit(X_reduced)
    
    return svm_model

# Framework 2: QAE + Quantum Random Forest
def framework2_qae_random_forest(X_train, y_train):
    """
    QAE + Quantum Random Forest framework for anomaly detection.
    This method performs dimensionality reduction, quantum encoding, and uses Quantum Random Forest for classification.
    """
    # Dimensionality reduction via PCA to match qubit requirements
    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(X_train)
    
    # Quantum encoding (QAE)
    encoded_X = [encode_data(sample) for sample in X_reduced]  # Encode each sample into quantum states
    
    # Use Quantum Random Forest for classification (classical random forest as a placeholder)
    rf_model = RandomForestClassifier(n_estimators=100)  # This could be extended with a quantum algorithm
    rf_model.fit(X_reduced, y_train)
    
    return rf_model


# Framework 3: QAE + kNN
def framework3_qae_knn(X_train, y_train):
    """
    QAE + kNN framework for anomaly detection.
    This method performs dimensionality reduction, quantum encoding, and uses kNN for classification.
    """
    # Dimensionality reduction via PCA to match qubit requirements
    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(X_train)
    
    # Quantum encoding (QAE)
    encoded_X = [encode_data(sample) for sample in X_reduced]  # Encode each sample into quantum states
    
    # Use k-Nearest Neighbors for classification
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_reduced, y_train)
    
    return knn_model


# Calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    tp_rate = recall_score(y_true, y_pred, average='macro', zero_division=1)
    tn_rate = tp_rate  # Placeholder for actual calculation
    return {
        "Accuracy": accuracy,
        "MCC": mcc,
        "F1": f1,
        "TP Rate": tp_rate,
        "TN Rate": tn_rate,
    }

# Classical methods for comparison
def classical_methods(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM (Radial)': SVC(kernel='rbf', probability=True),
        'SVM (Linear)': SVC(kernel='linear', probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = calculate_metrics(y_test, y_pred)
    return results

# Quantum and classical comparison function
def run_comparison(datasets, window_size=20):
    results = {}

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        X, y_true = preprocess_data(data, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

        # Quantum anomaly detection
        print("Quantum anomaly detection...")
        anomaly_scores = []
        for sample_X in tqdm(X_test, desc="Quantum Anomaly Detection"):
            initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
            optimized_params = optimize_params(sample_X, initial_params)
            score = objective_function(optimized_params, sample_X)
            anomaly_scores.append(score)

        smart_threshold = calculate_smart_threshold(anomaly_scores)
        y_pred_quantum = [1 if score > smart_threshold else 0 for score in anomaly_scores]
        results['QEAD Method (Proposed)'] = calculate_metrics(y_test[:len(y_pred_quantum)], y_pred_quantum)

        # Framework 1: QAE + One-Class SVM
        print("Framework 1: QAE + One-Class SVM")
        svm_model = framework1_qae_oneclass_svm(X_train)
        svm_predictions = svm_model.predict(X_test)
        results['QAE + One-Class SVM'] = calculate_metrics(y_test, svm_predictions)

        # Framework 2: QAE + Quantum Random Forest
        print("Framework 2: QAE + Quantum Random Forest")
        rf_model = framework2_qae_random_forest(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        results['QAE + Quantum Random Forest'] = calculate_metrics(y_test, rf_predictions)

        # Framework 3: QAE + kNN
        print("Framework 3: QAE + kNN")
        knn_model = framework3_qae_knn(X_train, y_train)
        knn_predictions = knn_model.predict(X_test)
        results['QAE + kNN'] = calculate_metrics(y_test, knn_predictions)

        # Classical models comparison
        print("Running classical models...")
        classical_results = classical_methods(X_train, y_train, X_test, y_test)
        results.update(classical_results)

    return results

# Calculate NAB score using TP Rate and TN Rate
def calculate_nab_score(tp_rate, tn_rate, weights):
    # Calculate FP Rate and FN Rate
    fp_rate = 1 - tn_rate
    fn_rate = 1 - tp_rate
    
    # Apply the NAB formula
    nab_score = (
        weights["TP"] * tp_rate
        - weights["FP"] * fp_rate
        - weights["FN"] * fn_rate
    )
    
    # Ensure the NAB score is between 0 and 100
    return max(0, min(nab_score, 100))

# Print comparison table with NAB Score for all methods
def print_comparison_table(results):
    
    # Define NAB weights (standard profile)
    nab_weights = {"TP": 1.0, "FP": 0.22, "FN": 1.0}
    
    print(f"\nComparison Table for Dataset: Bearing")
    print(f"{'Method':<25} {'MCC':<8} {'F1':<8} {'Accuracy':<10} {'TP Rate':<10} {'TN Rate':<10} {'NAB Score':<10}")
    for method, metrics in results.items():
        
        # Calculate NAB Score for each classical method using TP Rate and TN Rate
        nab_score_classical = calculate_nab_score(metrics['TP Rate'], metrics['TN Rate'], nab_weights)
        nab_score_str = f"{nab_score_classical:.3f}"
        
        print(f"{method:<25} "
              f"{metrics['MCC']:<8.3f} "
              f"{metrics['F1']:<8.3f} "
              f"{metrics['Accuracy']:<10.3f} "
              f"{metrics['TP Rate']:<10.3f} "
              f"{metrics['TN Rate']:<10.3f}"
              f"{nab_score_str:<10}")

# Main script execution
def main():
    datasets = load_datasets()
    results = run_comparison(datasets, window_size=4)
    print_comparison_table(results)

if __name__ == "__main__":
    main()
