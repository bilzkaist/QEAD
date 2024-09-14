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
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
import warnings

warnings.filterwarnings("ignore")

# Set dataset path for NAB
dataset_path = "/home/bilz/datasets/qead/NAB/data/"

# Use a non-interactive backend to avoid "Wayland" issues
import matplotlib
matplotlib.use('Agg')

# Preprocessing function for NAB dataset
def preprocess_data(data, window_size=20):
    print(f"Preprocessing NAB data with window size {window_size}...")

    if data.empty:
        raise ValueError("Dataset is empty.")

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    
    # Assume the dataset has a column for the metric to be used for anomaly detection
    data_column = data.columns[1]  # The second column typically contains the metric in NAB datasets
    
    data[data_column] = scaler.fit_transform(data[data_column].values.reshape(-1, 1))
    
    # Check if the label column exists
    if 'label' in data.columns:
        y_true = data['label'].values[window_size:]  # If the 'label' column exists, use it
    else:
        # If there's no label column, generate synthetic labels (e.g., all 0s for no anomaly)
        print("No 'label' column found. Generating synthetic labels (all non-anomalies)...")
        y_true = np.zeros(len(data) - window_size)  # Assuming no anomalies

    X = np.array([data[data_column].values[i:i + window_size] for i in range(len(data) - window_size)])
    return X, y_true


# Load dataset - Update to recursively search for CSV files in subdirectories
def load_datasets():
    print("Loading NAB datasets...")
    datasets = {}

    # Iterate through files and subdirectories in the NAB dataset folder
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                try:
                    data = pd.read_csv(file_path)
                    datasets[file_name] = data
                    print(f"Loaded {file_name} from {root}")
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
                    
    if not datasets:
        raise ValueError(f"No datasets loaded from path: {dataset_path}")

    return datasets

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

# Create and apply different noise models
def create_noise_models():
    noise_model = NoiseModel()

    # Depolarizing error
    depolarizing_error_1qubit = depolarizing_error(0.01, 1)
    depolarizing_error_2qubit = depolarizing_error(0.02, 2)
    
    # Amplitude damping error
    amplitude_damping = amplitude_damping_error(0.02)
    
    # Phase damping error
    phase_damping = phase_damping_error(0.01)
    
    # Adding noise to the noise model
    noise_model.add_all_qubit_quantum_error(depolarizing_error_1qubit, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error_2qubit, ['cx'])
    noise_model.add_all_qubit_quantum_error(amplitude_damping, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(phase_damping, ['u1', 'u2', 'u3'])

    # Display noise model for evaluation
    print("Applied Noise Model:")
    print(f"Depolarizing Error (1-Qubit): {depolarizing_error_1qubit}")
    print(f"Depolarizing Error (2-Qubit): {depolarizing_error_2qubit}")
    print(f"Amplitude Damping Error: {amplitude_damping}")
    print(f"Phase Damping Error: {phase_damping}")
    
    return noise_model

# Quantum and classical comparison function with noise support
def run_comparison_with_noise(datasets, window_size=20):
    results = {}

    # Create the noise models
    noise_model = create_noise_models()

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        X, y_true = preprocess_data(data, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

        # Quantum anomaly detection with noise
        anomaly_scores = []
        print(f"Running quantum anomaly detection with noise for dataset {name}...")
        for sample_X in tqdm(X_test, desc="Quantum Anomaly Detection with Noise"):
            initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
            optimized_params = optimize_params(sample_X, initial_params)  # No need for noise_model here
            score = objective_function(optimized_params, sample_X)
            anomaly_scores.append(score)

        smart_threshold = calculate_smart_threshold(anomaly_scores)
        y_pred_quantum = [1 if score > smart_threshold else 0 for score in anomaly_scores]
        quantum_metrics = calculate_metrics(y_test[:len(y_pred_quantum)], y_pred_quantum)

        # Classical models comparison
        classical_accuracies = {}
        models = classical_methods(X_train, y_train, X_test, y_test)
        print(f"Evaluating classical models for dataset {name}...")
        for model_name, (y_pred, y_pred_proba) in models.items():
            if y_pred_proba is not None:
                classical_accuracies[model_name] = calculate_metrics(y_test, y_pred)
                classical_accuracies[model_name]['ROC AUC'] = roc_auc_score(y_test, y_pred_proba)
                pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_pred_proba)
                classical_accuracies[model_name]['PR AUC'] = auc(pr_recall, pr_precision)
            else:
                classical_accuracies[model_name] = calculate_metrics(y_test, y_pred)
                classical_accuracies[model_name]['ROC AUC'] = "N/A"
                classical_accuracies[model_name]['PR AUC'] = "N/A"

        results[name] = {
            'quantum_metrics': quantum_metrics,
            'classical_accuracies': classical_accuracies
        }

    return results

# Calculate metrics function
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    tp_rate = recall_score(y_true, y_pred, average='macro', zero_division=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if confusion_matrix(y_true, y_pred).size == 4 else (0, 0, 0, 0)
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

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
        'Local Outlier Factor': LocalOutlierFactor(novelty=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # For models that support probability prediction
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability for class 1
            y_pred = model.predict(X_test)  # Hard predictions for calculating non-AUC metrics
            results[name] = (y_pred, y_pred_proba)
        else:
            y_pred = model.predict(X_test)
            results[name] = (y_pred, None)  # No probability predictions available

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
    nab_weights = {"TP": 1.0, "FP": 0.22, "FN": 1.0}

    for dataset_name, res in results.items():
        print(f"\nComparison Table for Dataset: {dataset_name}")
        print(f"{'Method':<25} {'MCC':<8} {'F1':<8} {'Accuracy':<10} {'TP Rate':<10} {'TN Rate':<10} {'PR AUC':<8} {'ROC AUC':<8} {'NAB Score':<10}")
        
        # Quantum method results
        quantum_metrics = res['quantum_metrics']
        pr_auc_str = quantum_metrics['PR AUC'] if isinstance(quantum_metrics['PR AUC'], str) else f"{quantum_metrics['PR AUC']:.3f}"
        roc_auc_str = quantum_metrics['ROC AUC'] if isinstance(quantum_metrics['ROC AUC'], str) else f"{quantum_metrics['ROC AUC']:.3f}"
        
        # Calculate NAB Score for Quantum Method using TP Rate and TN Rate
        nab_score_quantum = calculate_nab_score(quantum_metrics['TP Rate'], quantum_metrics['TN Rate'], nab_weights)
        nab_score_str = f"{nab_score_quantum:.3f}"
        
        print(f"{'Quantum Method':<25} "
              f"{quantum_metrics['MCC']:<8.3f} "
              f"{quantum_metrics['F1']:<8.3f} "
              f"{quantum_metrics['Accuracy']:<10.3f} "
              f"{quantum_metrics['TP Rate']:<10.3f} "
              f"{quantum_metrics['TN Rate']:<10.3f} "
              f"{pr_auc_str:<8} "
              f"{roc_auc_str:<8} "
              f"{nab_score_str:<10}")

        # Classical methods results
        for method, metrics in res['classical_accuracies'].items():
            pr_auc_str = metrics['PR AUC'] if isinstance(metrics['PR AUC'], str) else f"{metrics['PR AUC']:.3f}"
            roc_auc_str = metrics['ROC AUC'] if isinstance(metrics['ROC AUC'], str) else f"{metrics['ROC AUC']:.3f}"
            
            # Calculate NAB Score for each classical method using TP Rate and TN Rate
            nab_score_classical = calculate_nab_score(metrics['TP Rate'], metrics['TN Rate'], nab_weights)
            nab_score_str = f"{nab_score_classical:.3f}"
            
            print(f"{method:<25} "
                  f"{metrics['MCC']:<8.3f} "
                  f"{metrics['F1']:<8.3f} "
                  f"{metrics['Accuracy']:<10.3f} "
                  f"{metrics['TP Rate']:<10.3f} "
                  f"{metrics['TN Rate']:<10.3f} "
                  f"{pr_auc_str:<8} "
                  f"{roc_auc_str:<8} "
                  f"{nab_score_str:<10}")


# Main script execution
def main():
    datasets = load_datasets()
    #results = run_comparison(datasets, window_size=4)
    results = run_comparison_with_noise(datasets, window_size=4)
    print_comparison_table(results)

if __name__ == "__main__":
    main()
