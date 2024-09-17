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
from sklearn.model_selection import KFold
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

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    # Compute confusion matrix and handle multiclass/multilabel cases
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    # Initialize confusion matrix components
    tn, fp, fn, tp = 0, 0, 0, 0

    # Handle binary classification with one class present in y_true or y_pred
    if cm.shape == (1, 1):  # Only one class present in both y_true and y_pred
        if y_true[0] == 0:  # All negatives in y_true
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:  # All positives in y_true
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    elif cm.shape == (2, 1):  # Only negatives in y_pred
        tn, fp, fn, tp = cm[0, 0], 0, cm[1, 0], 0
    elif cm.shape == (1, 2):  # Only positives in y_pred
        tn, fp, fn, tp = 0, cm[0, 1], 0, cm[0, 0]
    elif cm.shape == (2, 2):  # Normal binary classification
        tn, fp, fn, tp = cm.ravel()

    # Calculate accuracy, MCC, F1 score
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0  # MCC only if more than one class
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    tp_rate = recall_score(y_true, y_pred, average='macro', zero_division=1)
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate PR AUC and ROC AUC if probabilities are provided
    if y_pred_proba is not None:
        if len(set(y_true)) > 2:  # Multiclass case
            pr_auc = "N/A"  # PR AUC for multiclass is not typically supported
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        else:  # Binary case
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
            results[name] = (y_pred, y_pred_proba)
        else:
            y_pred = model.predict(X_test)
            results[name] = (y_pred, None)  # No probability predictions available

    return results

# NAB Score Calculation Function
def calculate_nab_score2(detections, actual_anomalies, window_size, weights):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for detection in detections:
        detected_as_true_positive = False
        for (start, end) in actual_anomalies:
            if start <= detection <= end:
                true_positives += 1
                detected_as_true_positive = True
                break
        if not detected_as_true_positive:
            false_positives += 1

    for (start, end) in actual_anomalies:
        if not any(start <= detection <= end for detection in detections):
            false_negatives += 1

    nab_score = (
        weights["TP"] * true_positives
        - weights["FP"] * false_positives
        - weights["FN"] * false_negatives
    )

    nab_score = max(0, min(nab_score, 100))

    return nab_score

# Quantum and classical comparison function with 10-fold cross-validation
def run_comparison_10fold(datasets, window_size=20):
    results = {}
    kf = KFold(n_splits=10)

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        X, y_true = preprocess_data(data, window_size)

        # Store metrics across all folds
        qead_results = []
        classical_results = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]

            # Quantum anomaly detection
            anomaly_scores = []
            for sample_X in tqdm(X_test, desc="Quantum Anomaly Detection"):
                initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
                optimized_params = optimize_params(sample_X, initial_params)
                score = objective_function(optimized_params, sample_X)
                anomaly_scores.append(score)

            smart_threshold = calculate_smart_threshold(anomaly_scores)
            y_pred_quantum = [1 if score > smart_threshold else 0 for score in anomaly_scores]
            qead_results.append(calculate_metrics(y_test[:len(y_pred_quantum)], y_pred_quantum))

            # Classical models comparison
            classical_results.append(classical_methods(X_train, y_train, X_test, y_test))

        # Average the results across folds
        results[name] = {
            'quantum_metrics': average_metrics(qead_results),
            'classical_accuracies': average_classical_metrics(classical_results)
        }

    return results

# Function to average metrics across folds
# Function to average metrics across folds
def average_metrics(fold_metrics):
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        # Extract numeric values only, ignoring strings like "N/A"
        numeric_values = [metrics[key] for metrics in fold_metrics if isinstance(metrics[key], (int, float))]
        
        # Check if there are numeric values to average, otherwise assign "N/A"
        if numeric_values:
            avg_metrics[key] = np.mean(numeric_values)
        else:
            avg_metrics[key] = "N/A"  # If no numeric values, set the result as "N/A"
    return avg_metrics


# Function to average classical results across folds
def average_classical_metrics(fold_classical_results):
    avg_classical = {}
    for method in fold_classical_results[0]:
        avg_classical[method] = average_metrics([fold[method] for fold in fold_classical_results])
    return avg_classical

# Calculate NAB score using TP Rate and TN Rate
def calculate_nab_score(tp_rate, tn_rate, weights):
    fp_rate = 1 - tn_rate
    fn_rate = 1 - tp_rate
    
    nab_score = (
        weights["TP"] * tp_rate
        - weights["FP"] * fp_rate
        - weights["FN"] * fn_rate
    )
    
    return max(0, min(nab_score, 100))

# Print comparison table with NAB Score for all methods
def print_comparison_table(results):
    nab_weights = {"TP": 1.0, "FP": 0.22, "FN": 1.0}

    for dataset_name, res in results.items():
        print(f"\nComparison Table for Dataset: {dataset_name}")
        print(f"{'Method':<25} {'MCC':<8} {'F1':<8} {'Accuracy':<10} {'TP Rate':<10} {'TN Rate':<10} {'PR AUC':<8} {'ROC AUC':<8} {'NAB Score':<10}")
        
        quantum_metrics = res['quantum_metrics']
        pr_auc_str = quantum_metrics['PR AUC'] if isinstance(quantum_metrics['PR AUC'], str) else f"{quantum_metrics['PR AUC']:.3f}"
        roc_auc_str = quantum_metrics['ROC AUC'] if isinstance(quantum_metrics['ROC AUC'], str) else f"{quantum_metrics['ROC AUC']:.3f}"
        
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

        for method, metrics in res['classical_accuracies'].items():
            pr_auc_str = metrics['PR AUC'] if isinstance(metrics['PR AUC'], str) else f"{metrics['PR AUC']:.3f}"
            roc_auc_str = metrics['ROC AUC'] if isinstance(metrics['ROC AUC'], str) else f"{metrics['ROC AUC']:.3f}"
            
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
    results = run_comparison_10fold(datasets, window_size=4)
    print_comparison_table(results)

if __name__ == "__main__":
    main()
