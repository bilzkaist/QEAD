import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from qiskit.exceptions import QiskitError
import matplotlib
matplotlib.use('Agg')
import warnings

# Ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Set dataset path and file name for NF-UNSW-NB15 dataset
dataset_path = "/home/bilz/datasets/qead/"
dataset_name = "NF-UNSW-NB15.csv"

# Preprocessing function for NF-UNSW-NB15 dataset
def preprocess_data(data, qubit_no=20, chunk_size=10000):
    print(f"Preprocessing NF-UNSW-NB15 data with qubit no {qubit_no}...")

    if data.empty:
        raise ValueError("Dataset is empty.")
    
    # Encoding Attack labels to 0 (Benign) and 1 (Attack)
    label_encoder = LabelEncoder()
    data['Attack'] = label_encoder.fit_transform(data['Attack'])

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    
    # Select relevant columns
    features = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS']
    
    # Scale the selected features
    data[features] = scaler.fit_transform(data[features])
    
    # Process the dataset in chunks
    X = []
    y_true = []
    for i in range(0, len(data) - qubit_no, chunk_size):
        chunk = data[i:i + chunk_size + qubit_no]
        if len(chunk) < qubit_no:
            break
        
        X_chunk = np.array([chunk[features].values[j:j + qubit_no] for j in range(len(chunk) - qubit_no)])
        y_chunk = np.array(chunk['Attack'][qubit_no:])
        
        X.append(X_chunk)
        y_true.append(y_chunk)

    X = np.concatenate(X, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    
    print("Preprocessing complete.")
    return X, y_true

# Function to load the NF-UNSW-NB15 dataset from the local CSV file
def load_datasets():
    print("Loading NF-UNSW-NB15 dataset from local CSV file...")
    datasets = {}

    # Combine the path and name into a full file path
    file_path = os.path.join(dataset_path, dataset_name)
    data_nab = pd.read_csv(file_path)
    
    print(f"Columns in the dataset: {data_nab.columns}")
    
    if data_nab.empty:
        raise ValueError("Failed to load dataset.")
    
    datasets['NF-UNSW-NB15'] = data_nab
    print("Loaded NF-UNSW-NB15 dataset.")
    return datasets

# Quantum encoding function
def encode_data(X):
    """Encodes classical data into a quantum state."""
    n_qubits = len(X.flatten())  # Ensure number of qubits matches data length
    qc = QuantumCircuit(n_qubits)
    
    flat_X = X.flatten()  # Flatten the array if multi-dimensional
    for i, x in enumerate(flat_X):
        qc.ry(float(x), i)  # Ensure that 'x' is a scalar using float()
    
    return qc

# Variational circuit function
def variational_circuit(params, n_qubits):
    """Creates a variational circuit."""
    qc = QuantumCircuit(n_qubits)
    for i, param in enumerate(params):
        qc.rx(float(param), i)  # Convert param to float
        qc.rz(float(param), i)  # Convert param to float
    return qc

# Objective function for quantum method
loss_history = []
def objective_function(params, X):
    """Objective function to minimize in the quantum approach."""
    n_qubits = len(X.flatten())  # Flatten the input if multi-dimensional
    qc = encode_data(X)
    var_circuit = variational_circuit(params, n_qubits)
    qc.compose(var_circuit, inplace=True)
    
    # Add instruction to save the statevector
    qc.save_statevector()

    # Use AerSimulator with statevector method
    simulator = AerSimulator(method='statevector')
    transpiled_qc = transpile(qc, simulator)

    try:
        # Run the circuit
        result = simulator.run(transpiled_qc).result()
        
        # Retrieve the statevector from the result
        statevector = np.asarray(result.get_statevector(transpiled_qc))
    except Exception as e:
        raise QiskitError(f"No statevector available: {str(e)}")
    
    # Define the normal state for comparison
    normal_state = np.zeros_like(statevector)
    normal_state[0] = 1

    # Calculate the loss as 1 - the overlap between the state and the normal state
    loss = 1 - np.abs(np.dot(statevector.conj(), normal_state)) ** 2

    # Append the loss to the history for tracking
    loss_history.append(loss)

    return loss

# Optimize quantum circuit parameters
def optimize_params(X, initial_params):
    """Optimizes the parameters of the quantum circuit."""
    print("Optimizing quantum circuit parameters...")
    result = minimize(objective_function, initial_params, args=(X,), method='COBYLA')
    print("Optimization complete.")
    return result.x

# Classical anomaly detection models
def classical_methods(X_train, y_train, X_test, y_test):
    """Evaluates classical methods for anomaly detection."""
    models = {
        'IsolationForest': IsolationForest(contamination=0.1),
        'OneClassSVM': OneClassSVM(nu=0.1),
        'LocalOutlierFactor': LocalOutlierFactor(novelty=True),
    }
    
    results = {}
    for name, model in models.items():
        print(f"Running {name}...")
        model.fit(X_train.reshape(len(X_train), -1))
        y_pred = model.predict(X_test.reshape(len(X_test), -1))
        y_pred = np.where(y_pred == 1, 0, 1)  # Convert 1 -> 0 (normal), -1 -> 1 (anomaly)
        accuracy = accuracy_score(y_test[:len(y_pred)], y_pred)
        print(f"{name} accuracy: {accuracy:.2f}")
        results[name] = accuracy
    
    return results

# Smart threshold using IQR method
def calculate_smart_threshold(scores):
    """Calculates a smart threshold based on IQR."""
    q25, q75 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    return upper_bound

# Main function to run the comparison
def run_comparison(datasets, qubit_no=20):
    results = {}

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        X, y_true = preprocess_data(data, qubit_no)

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)
        
        # Quantum anomaly detection
        anomaly_scores = []
        print(f"Running quantum anomaly detection for {name}...")
        for sample_X in tqdm(X_test, desc="Quantum Anomaly Detection"):
            initial_params = np.random.rand(len(sample_X.flatten())) * 2 * np.pi
            optimized_params = optimize_params(sample_X, initial_params)
            score = objective_function(optimized_params, sample_X)
            anomaly_scores.append(score)

        # Smart threshold based on IQR
        smart_threshold = calculate_smart_threshold(anomaly_scores)
        y_pred_quantum = [1 if score > smart_threshold else 0 for score in anomaly_scores]

        # Quantum method accuracy
        quantum_accuracy = accuracy_score(y_test[:len(y_pred_quantum)], y_pred_quantum)
        print(f"Quantum Method Accuracy for {name}: {quantum_accuracy:.2f}")

        # Classical anomaly detection
        classical_accuracies = classical_methods(X_train, y_train, X_test, y_test)

        # Store results for visualization
        results[name] = {
            'quantum_accuracy': quantum_accuracy,
            'classical_accuracies': classical_accuracies,
            'y_test': y_test,
            'y_pred_quantum': y_pred_quantum,
            'anomaly_scores': anomaly_scores,
            'loss_history': loss_history.copy(),
            'threshold': smart_threshold
        }

        # Clear loss history for the next dataset
        loss_history.clear()

    return results

# Main script execution
def main():
    datasets = load_datasets()
    results = run_comparison(datasets, qubit_no=4)
    print(f"Results saved.")

if __name__ == "__main__":
    main()