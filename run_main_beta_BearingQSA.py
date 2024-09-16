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

# Variational circuit function (for QEAD)
def variational_circuit(params, n_qubits):
    qc = QuantumCircuit(n_qubits)
    for i, param in enumerate(params):
        qc.rx(param, i)
        qc.rz(param, i)
    return qc

# Quantum anomaly detection objective function (for QEAD)
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

# Optimize quantum circuit parameters (for QEAD)
def optimize_params(X, initial_params):
    result = minimize(objective_function, initial_params, args=(X,), method='COBYLA')
    return result.x

# Smart threshold using IQR
def calculate_smart_threshold(scores):
    q25, q75 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    return upper_bound

# Quantum Self-Attention (QSA) model class
class QuantumSelfAttention:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.qc = QuantumCircuit(self.num_inputs)

    # Method to encode classical inputs as quantum states
    def encode_inputs(self, inputs):
        for i, value in enumerate(inputs):
            self.qc.ry(value, i)

    # Apply ansatz circuit for Query, Key, Value generation
    def apply_ansatz(self):
        for i in range(self.num_inputs):
            self.qc.rx(np.pi/4, i)

    # Measurement of query/key
    def measure_attention(self):
        self.qc.measure_all()

    # Simulate the quantum attention circuit
    def simulate(self, noise_model=None):
        simulator = AerSimulator(noise_model=noise_model)
        job = simulator.run(self.qc)
        result = job.result()
        counts = result.get_counts(self.qc)
        return counts

    # Run the full quantum self-attention circuit
    def run(self, inputs, noise_model=None):
        self.encode_inputs(inputs)
        self.apply_ansatz()
        self.measure_attention()
        return self.simulate(noise_model=noise_model)

# Process the output from quantum self-attention circuit to get anomaly scores
def process_attention_output(result):
    counts = list(result.values())
    probabilities = np.array(counts) / sum(counts)
    score = 1 - probabilities[0] if len(probabilities) > 0 else 0  # Handle case when probabilities are empty
    return score

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    # Compute confusion matrix and handle multiclass/multilabel cases
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    # Initialize confusion matrix components
    tn, fp, fn, tp = 0, 0, 0, 0

    # Check the shape of the confusion matrix and handle it accordingly
    if cm.shape == (2, 2):  # Binary classification with both classes present
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape[0] == 1:  # Only one class present in y_true
        if y_true[0] == 0:  # All negatives in y_true
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], 0, 0
        else:  # All positives in y_true
            tn, fp, fn, tp = 0, 0, cm[0, 0], cm[0, 1]
    elif cm.shape[1] == 1:  # Only one class present in y_pred
        tn, fp, fn, tp = cm[0, 0], 0, cm[1, 0], 0
    else:  # Handle cases where there are more than two classes
        raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

    # Calculate accuracy, MCC, F1 score
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0  # MCC only if more than one class
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    tp_rate = recall_score(y_true, y_pred, average='macro', zero_division=1)
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate PR AUC and ROC AUC if probabilities are provided
    if y_pred_proba is not None:
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(pr_recall, pr_precision)
        roc_auc = roc_auc_score(y_true, y_pred_proba, average='macro')
    else:
        pr_auc = "N/A"
        roc_auc = "N/A"

    return {
        "Accuracy": accuracy,
        "MCC": mcc,
        "F1": f1,
        "TP Rate": tp_rate,
        "TN Rate": tn_rate,
        "PR AUC": pr_auc,
        "ROC AUC": roc_auc
    }

# Classical anomaly detection models
def classical_methods(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Linear Regression': LogisticRegression(),
        'Naive Bayes': GaussianNB(),
        'SVM (Radial)': SVC(kernel='rbf', probability=True),
        'SVM (Linear)': SVC(kernel='linear', probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
        'Robust Covariance': EllipticEnvelope(),
        'Isolation Forest': IsolationForest(),
        'One-Class SVM': OneClassSVM(),
        'Local Outlier Factor': LocalOutlierFactor(novelty=True)  # novelty=True allows predict method
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability for class 1
            y_pred = model.predict(X_test)
            results[name] = calculate_metrics(y_test, y_pred, y_pred_proba)
        else:
            y_pred = model.predict(X_test)
            results[name] = calculate_metrics(y_test, y_pred)

    return results

# Quantum and classical comparison function with NAB Score for QEAD, QSA, and classical models
def run_comparison(datasets, window_size=20):
    results = {}

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        X, y_true = preprocess_data(data, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

        # Quantum Enhanced Anomaly Detection (QEAD)
        anomaly_scores_qead = []
        print(f"Running QEAD for dataset {name}...")
        for sample_X in tqdm(X_test, desc="QEAD"):
            initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
            optimized_params = optimize_params(sample_X, initial_params)
            score = objective_function(optimized_params, sample_X)
            anomaly_scores_qead.append(score)
        
        smart_threshold_qead = calculate_smart_threshold(anomaly_scores_qead)
        y_pred_qead = [1 if score > smart_threshold_qead else 0 for score in anomaly_scores_qead]
        quantum_metrics_qead = calculate_metrics(y_test[:len(y_pred_qead)], y_pred_qead)

        # Quantum Self-Attention (QSA)
        print(f"Running Quantum Self-Attention for dataset {name}...")
        qsa = QuantumSelfAttention(window_size)
        anomaly_scores_qsa = []
        for sample_X in tqdm(X_test, desc="Quantum Self-Attention"):
            qsa_results = qsa.run(sample_X)
            score = process_attention_output(qsa_results)
            anomaly_scores_qsa.append(score)

        smart_threshold_qsa = calculate_smart_threshold(anomaly_scores_qsa)
        y_pred_qsa = [1 if score > smart_threshold_qsa else 0 for score in anomaly_scores_qsa]
        quantum_metrics_qsa = calculate_metrics(y_test[:len(y_pred_qsa)], y_pred_qsa)

        # Classical models comparison
        classical_accuracies = classical_methods(X_train, y_train, X_test, y_test)

        results[name] = {
            'quantum_metrics_qead': quantum_metrics_qead,
            'quantum_metrics_qsa': quantum_metrics_qsa,
            'classical_accuracies': classical_accuracies
        }

    return results

# Print comparison table with NAB Score for QEAD, QSA, and classical models
def print_comparison_table(results):
    nab_weights = {"TP": 1.0, "FP": 0.22, "FN": 1.0}

    for dataset_name, res in results.items():
        print(f"\nComparison Table for Dataset: {dataset_name}")
        print(f"{'Method':<25} {'MCC':<8} {'F1':<8} {'Accuracy':<10} {'TP Rate':<10} {'TN Rate':<10} {'PR AUC':<8} {'ROC AUC':<8} {'NAB Score':<10}")
        
        # QEAD results
        quantum_metrics_qead = res['quantum_metrics_qead']
        pr_auc_str = quantum_metrics_qead['PR AUC'] if isinstance(quantum_metrics_qead['PR AUC'], str) else f"{quantum_metrics_qead['PR AUC']:.3f}"
        roc_auc_str = quantum_metrics_qead['ROC AUC'] if isinstance(quantum_metrics_qead['ROC AUC'], str) else f"{quantum_metrics_qead['ROC AUC']:.3f}"
        print(f"{'Quantum Method (QEAD)':<25} "
              f"{quantum_metrics_qead['MCC']:<8.3f} "
              f"{quantum_metrics_qead['F1']:<8.3f} "
              f"{quantum_metrics_qead['Accuracy']:<10.3f} "
              f"{quantum_metrics_qead['TP Rate']:<10.3f} "
              f"{quantum_metrics_qead['TN Rate']:<10.3f} "
              f"{pr_auc_str:<8} "
              f"{roc_auc_str:<8} ")

        # QSA results
        quantum_metrics_qsa = res['quantum_metrics_qsa']
        pr_auc_str_qsa = quantum_metrics_qsa['PR AUC'] if isinstance(quantum_metrics_qsa['PR AUC'], str) else f"{quantum_metrics_qsa['PR AUC']:.3f}"
        roc_auc_str_qsa = quantum_metrics_qsa['ROC AUC'] if isinstance(quantum_metrics_qsa['ROC AUC'], str) else f"{quantum_metrics_qsa['ROC AUC']:.3f}"
        print(f"{'Quantum Method (QSA)':<25} "
              f"{quantum_metrics_qsa['MCC']:<8.3f} "
              f"{quantum_metrics_qsa['F1']:<8.3f} "
              f"{quantum_metrics_qsa['Accuracy']:<10.3f} "
              f"{quantum_metrics_qsa['TP Rate']:<10.3f} "
              f"{quantum_metrics_qsa['TN Rate']:<10.3f} "
              f"{pr_auc_str_qsa:<8} "
              f"{roc_auc_str_qsa:<8} ")

        # Classical methods results
        for method, metrics in res['classical_accuracies'].items():
            pr_auc_str = metrics['PR AUC'] if isinstance(metrics['PR AUC'], str) else f"{metrics['PR AUC']:.3f}"
            roc_auc_str = metrics['ROC AUC'] if isinstance(metrics['ROC AUC'], str) else f"{metrics['ROC AUC']:.3f}"
            print(f"{method:<25} "
                  f"{metrics['MCC']:<8.3f} "
                  f"{metrics['F1']:<8.3f} "
                  f"{metrics['Accuracy']:<10.3f} "
                  f"{metrics['TP Rate']:<10.3f} "
                  f"{metrics['TN Rate']:<10.3f} "
                  f"{pr_auc_str:<8} "
                  f"{roc_auc_str:<8} ")

# Main script execution
def main():
    datasets = load_datasets()
    results = run_comparison(datasets, window_size=4)
    print_comparison_table(results)

if __name__ == "__main__":
    main()
