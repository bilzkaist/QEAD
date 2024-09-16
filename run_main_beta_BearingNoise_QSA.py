import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit import QuantumCircuit
import warnings

warnings.filterwarnings("ignore")

# Set dataset path and file name for the Bearing dataset
dataset_path = "/home/bilz/datasets/qead/q/"
dataset_name = "Bearing.csv"

# Preprocessing function for Bearing dataset
def preprocess_data(data, qubit_no=20):
    if data.empty:
        raise ValueError("Dataset is empty.")

    scaler = MinMaxScaler(feature_range=(0, np.pi))

    if 'Bearing 1' in data.columns:
        data['Bearing 1'] = scaler.fit_transform(data['Bearing 1'].values.reshape(-1, 1))
        X = np.array([data['Bearing 1'].values[i:i + qubit_no] for i in range(len(data) - qubit_no)])
        y_true = np.array([1 if val > 0.8 else 0 for val in data['Bearing 1'][qubit_no:]])
    else:
        raise ValueError("Unknown data format in dataset.")

    return X, y_true

# Load dataset
def load_datasets():
    file_path = dataset_path + dataset_name
    data_Bearing = pd.read_csv(file_path)
    return {'Bearing': data_Bearing}

# Quantum encoding function with noise model application
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
def objective_function(params, X, noise_model=None):
    n_qubits = len(X)
    qc = encode_data(X)
    var_circuit = variational_circuit(params, n_qubits)
    qc.compose(var_circuit, inplace=True)
    qc.measure_all()

    simulator = AerSimulator(noise_model=noise_model)
    job = simulator.run(qc)
    result = job.result()
    counts = result.get_counts(qc)
    probability_of_0 = counts.get('0' * n_qubits, 0) / sum(counts.values())
    loss = 1 - probability_of_0
    
    return loss

# Function to optimize the quantum circuit parameters
def optimize_params(X, initial_params, noise_model=None):
    result = minimize(objective_function, initial_params, args=(X, noise_model), method='COBYLA')
    return result.x

# Quantum Self Attention (QSA) for anomaly detection
class QuantumSelfAttention:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.qc = QuantumCircuit(self.num_inputs)

    def encode_inputs(self, inputs):
        for i, value in enumerate(inputs):
            self.qc.ry(value, i)

    def apply_ansatz(self):
        for i in range(self.num_inputs):
            self.qc.rx(np.pi/4, i)

    def measure_attention(self):
        self.qc.measure_all()

    def simulate(self, noise_model=None):
        simulator = AerSimulator(noise_model=noise_model)
        job = simulator.run(self.qc)
        result = job.result()
        counts = result.get_counts(self.qc)
        return counts

    def run(self, inputs, noise_model=None):
        self.encode_inputs(inputs)
        self.apply_ansatz()
        self.measure_attention()
        return self.simulate(noise_model=noise_model)

# Quantum Self-Attention anomaly detection
def quantum_self_attention_anomaly_detection(X, noise_model=None):
    anomaly_scores = []
    for sample_X in tqdm(X, desc="Quantum Self-Attention Anomaly Detection"):
        qsa = QuantumSelfAttention(len(sample_X))
        result = qsa.run(sample_X, noise_model)
        score = process_attention_output(result)
        anomaly_scores.append(score)
    return anomaly_scores

# Process the output from quantum self-attention circuit to get anomaly scores
def process_attention_output(result):
    counts = list(result.values())
    probabilities = np.array(counts) / sum(counts)
    score = 1 - probabilities[0]
    return score

# Create and apply different noise models
def create_noise_models():
    noise_model = NoiseModel()
    depolarizing_error_1qubit = depolarizing_error(0.01, 1)
    depolarizing_error_2qubit = depolarizing_error(0.02, 2)
    amplitude_damping = amplitude_damping_error(0.02)
    phase_damping = phase_damping_error(0.01)
    
    noise_model.add_all_qubit_quantum_error(depolarizing_error_1qubit, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error_2qubit, ['cx'])
    noise_model.add_all_qubit_quantum_error(amplitude_damping, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(phase_damping, ['u1', 'u2', 'u3'])
    
    return noise_model

# Classical anomaly detection models
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
        'Isolation Forest': IsolationForest(),
        'One-Class SVM': OneClassSVM()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = calculate_metrics(y_test, y_pred)
    return results

# Calculate metrics function
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0
    f1 = f1_score(y_true, y_pred, average='macro')
    tp_rate = recall_score(y_true, y_pred, average='macro')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if confusion_matrix(y_true, y_pred).size == 4 else (0, 0, 0, 0)
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "Accuracy": accuracy,
        "MCC": mcc,
        "F1": f1,
        "TP Rate": tp_rate,
        "TN Rate": tn_rate,
    }

# Smart threshold using IQR
def calculate_smart_threshold(scores):
    q25, q75 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    return upper_bound

# Run comparison with and without noise
def run_comparison_with_and_without_noise(datasets, qubit_no=20):
    results = {}
    noise_model = create_noise_models()

    for name, data in datasets.items():
        X, y_true = preprocess_data(data, qubit_no)
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

        # Quantum Enhanced Anomaly Detection (QEAD) with and without noise
        anomaly_scores_qead_noise = []
        anomaly_scores_qead_no_noise = []
        
        for sample_X in tqdm(X_test, desc="QEAD with Noise"):
            initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
            optimized_params = optimize_params(sample_X, initial_params, noise_model=noise_model)
            score_with_noise = objective_function(optimized_params, sample_X, noise_model=noise_model)
            score_without_noise = objective_function(optimized_params, sample_X, noise_model=None)
            
            anomaly_scores_qead_noise.append(score_with_noise)
            anomaly_scores_qead_no_noise.append(score_without_noise)
        
        # Quantum Self-Attention (QSA) with and without noise
        anomaly_scores_qsa_noise = quantum_self_attention_anomaly_detection(X_test, noise_model)
        anomaly_scores_qsa_no_noise = quantum_self_attention_anomaly_detection(X_test, noise_model=None)

        # Apply threshold for anomaly detection
        smart_threshold_qead_noise = calculate_smart_threshold(anomaly_scores_qead_noise)
        y_pred_qead_noise = [1 if score > smart_threshold_qead_noise else 0 for score in anomaly_scores_qead_noise]

        smart_threshold_qead_no_noise = calculate_smart_threshold(anomaly_scores_qead_no_noise)
        y_pred_qead_no_noise = [1 if score > smart_threshold_qead_no_noise else 0 for score in anomaly_scores_qead_no_noise]

        smart_threshold_qsa_noise = calculate_smart_threshold(anomaly_scores_qsa_noise)
        y_pred_qsa_noise = [1 if score > smart_threshold_qsa_noise else 0 for score in anomaly_scores_qsa_noise]

        smart_threshold_qsa_no_noise = calculate_smart_threshold(anomaly_scores_qsa_no_noise)
        y_pred_qsa_no_noise = [1 if score > smart_threshold_qsa_no_noise else 0 for score in anomaly_scores_qsa_no_noise]

        # Calculate metrics for QEAD and QSA with and without noise
        quantum_metrics_qead_noise = calculate_metrics(y_test[:len(y_pred_qead_noise)], y_pred_qead_noise)
        quantum_metrics_qead_no_noise = calculate_metrics(y_test[:len(y_pred_qead_no_noise)], y_pred_qead_no_noise)
        quantum_metrics_qsa_noise = calculate_metrics(y_test[:len(y_pred_qsa_noise)], y_pred_qsa_noise)
        quantum_metrics_qsa_no_noise = calculate_metrics(y_test[:len(y_pred_qsa_no_noise)], y_pred_qsa_no_noise)

        # Classical methods comparison
        classical_accuracies = classical_methods(X_train, y_train, X_test, y_test)

        results[name] = {
            'quantum_metrics_qead_noise': quantum_metrics_qead_noise,
            'quantum_metrics_qead_no_noise': quantum_metrics_qead_no_noise,
            'quantum_metrics_qsa_noise': quantum_metrics_qsa_noise,
            'quantum_metrics_qsa_no_noise': quantum_metrics_qsa_no_noise,
            'classical_accuracies': classical_accuracies
        }

    return results

# Print comparison table with NAB Score for all methods
def print_comparison_table(results):
    nab_weights = {"TP": 1.0, "FP": 0.22, "FN": 1.0}

    for dataset_name, res in results.items():
        print(f"\nComparison Table for Dataset: {dataset_name}")
        print(f"{'Method':<30} {'MCC':<8} {'F1':<8} {'Accuracy':<10} {'TP Rate':<10} {'TN Rate':<10} {'NAB Score':<10}")

        # Print QEAD results with and without noise
        quantum_metrics_qead_noise = res['quantum_metrics_qead_noise']
        quantum_metrics_qead_no_noise = res['quantum_metrics_qead_no_noise']
        nab_score_qead_noise = calculate_nab_score(quantum_metrics_qead_noise['TP Rate'], quantum_metrics_qead_noise['TN Rate'], nab_weights)
        nab_score_qead_no_noise = calculate_nab_score(quantum_metrics_qead_no_noise['TP Rate'], quantum_metrics_qead_no_noise['TN Rate'], nab_weights)

        print(f"{'Quantum Method (QEAD with Noise)':<30} "
              f"{quantum_metrics_qead_noise['MCC']:<8.3f} "
              f"{quantum_metrics_qead_noise['F1']:<8.3f} "
              f"{quantum_metrics_qead_noise['Accuracy']:<10.3f} "
              f"{quantum_metrics_qead_noise['TP Rate']:<10.3f} "
              f"{quantum_metrics_qead_noise['TN Rate']:<10.3f} "
              f"{nab_score_qead_noise:<10.3f}")

        print(f"{'Quantum Method (QEAD no Noise)':<30} "
              f"{quantum_metrics_qead_no_noise['MCC']:<8.3f} "
              f"{quantum_metrics_qead_no_noise['F1']:<8.3f} "
              f"{quantum_metrics_qead_no_noise['Accuracy']:<10.3f} "
              f"{quantum_metrics_qead_no_noise['TP Rate']:<10.3f} "
              f"{quantum_metrics_qead_no_noise['TN Rate']:<10.3f} "
              f"{nab_score_qead_no_noise:<10.3f}")

        # Print QSA results with and without noise
        quantum_metrics_qsa_noise = res['quantum_metrics_qsa_noise']
        quantum_metrics_qsa_no_noise = res['quantum_metrics_qsa_no_noise']
        nab_score_qsa_noise = calculate_nab_score(quantum_metrics_qsa_noise['TP Rate'], quantum_metrics_qsa_noise['TN Rate'], nab_weights)
        nab_score_qsa_no_noise = calculate_nab_score(quantum_metrics_qsa_no_noise['TP Rate'], quantum_metrics_qsa_no_noise['TN Rate'], nab_weights)

        print(f"{'Quantum Method (QSA with Noise)':<30} "
              f"{quantum_metrics_qsa_noise['MCC']:<8.3f} "
              f"{quantum_metrics_qsa_noise['F1']:<8.3f} "
              f"{quantum_metrics_qsa_noise['Accuracy']:<10.3f} "
              f"{quantum_metrics_qsa_noise['TP Rate']:<10.3f} "
              f"{quantum_metrics_qsa_noise['TN Rate']:<10.3f} "
              f"{nab_score_qsa_noise:<10.3f}")

        print(f"{'Quantum Method (QSA no Noise)':<30} "
              f"{quantum_metrics_qsa_no_noise['MCC']:<8.3f} "
              f"{quantum_metrics_qsa_no_noise['F1']:<8.3f} "
              f"{quantum_metrics_qsa_no_noise['Accuracy']:<10.3f} "
              f"{quantum_metrics_qsa_no_noise['TP Rate']:<10.3f} "
              f"{quantum_metrics_qsa_no_noise['TN Rate']:<10.3f} "
              f"{nab_score_qsa_no_noise:<10.3f}")

        # Print Classical methods results
        for method, metrics in res['classical_accuracies'].items():
            nab_score_classical = calculate_nab_score(metrics['TP Rate'], metrics['TN Rate'], nab_weights)
            print(f"{method:<30} "
                  f"{metrics['MCC']:<8.3f} "
                  f"{metrics['F1']:<8.3f} "
                  f"{metrics['Accuracy']:<10.3f} "
                  f"{metrics['TP Rate']:<10.3f} "
                  f"{metrics['TN Rate']:<10.3f} "
                  f"{nab_score_classical:<10.3f}")

# NAB score calculation
def calculate_nab_score(tp_rate, tn_rate, weights):
    fp_rate = 1 - tn_rate
    fn_rate = 1 - tp_rate
    nab_score = weights["TP"] * tp_rate - weights["FP"] * fp_rate - weights["FN"] * fn_rate
    return max(0, min(nab_score, 100))

# Main script execution
def main():
    datasets = load_datasets()
    results = run_comparison_with_and_without_noise(datasets, qubit_no=4)
    print_comparison_table(results)

if __name__ == "__main__":
    main()
