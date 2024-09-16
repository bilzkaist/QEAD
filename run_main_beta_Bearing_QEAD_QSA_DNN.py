import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
import torch.utils.data as data

# Set dataset path and file name for the Bearing dataset
dataset_path = "/home/bilz/datasets/qead/q/"
dataset_name = "Bearing.csv"

# Use a non-interactive backend to avoid "Wayland" issues
import matplotlib
matplotlib.use('Agg')

# Define the BearingDataset class for PyTorch
class BearingDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

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

# Define Quantum Self-Attention class (QSA)
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
    def process_attention_output(self, result):
        counts = list(result.values())
        probabilities = np.array(counts) / sum(counts)
        score = 1 - probabilities[0]
        return score

# Define Self-Attention DNN Model in PyTorch
class SelfAttentionDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, output_size):
        super(SelfAttentionDNN, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(0)  # Add batch dimension for attention
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(0)  # Remove batch dimension
        x = self.relu(self.fc1(attn_output))
        x = self.fc2(x)
        return x

# Training Function for DNN
def train_dnn_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluate DNN Model
def evaluate_dnn_model(model, test_loader, device):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int)
            y_pred.extend(predictions.flatten())
            y_true.extend(labels.numpy())

    return y_true, y_pred

# DNN-based anomaly detection
def dnn_anomaly_detection(X_train, y_train, X_test, y_test, device, batch_size=16, epochs=10):
    train_dataset = BearingDataset(X_train, y_train)
    test_dataset = BearingDataset(X_test, y_test)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    input_size = X_train.shape[1]
    hidden_size = 128
    num_heads = 4
    output_size = 1

    model = SelfAttentionDNN(input_size=input_size, hidden_size=hidden_size, num_heads=num_heads, output_size=output_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dnn_model(model, train_loader, criterion, optimizer, device, epochs=epochs)

    y_true, y_pred = evaluate_dnn_model(model, test_loader, device)
    
    return calculate_metrics(y_true, y_pred)

# NAB Score Calculation Function
def calculate_nab_score(tp_rate, tn_rate, weights):
    fp_rate = 1 - tn_rate
    fn_rate = 1 - tp_rate
    
    nab_score = (
        weights["TP"] * tp_rate
        - weights["FP"] * fp_rate
        - weights["FN"] * fn_rate
    )
    
    return max(0, min(nab_score, 100))

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    tp_rate = recall_score(y_true, y_pred, average='macro', zero_division=1)
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Accuracy": accuracy,
        "MCC": mcc,
        "F1": f1,
        "TP Rate": tp_rate,
        "TN Rate": tn_rate
    }

# Define run_comparison function with NAB Score calculation
def run_comparison(datasets, window_size=20, device='cpu'):
    nab_weights = {"TP": 1.0, "FP": 0.22, "FN": 1.0}
    results = {}

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        X, y_true = preprocess_data(data, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

        # Quantum Enhanced Anomaly Detection (QEAD)
        print(f"Running QEAD for dataset {name}...")
        anomaly_scores_qead = []
        for sample_X in tqdm(X_test, desc="QEAD"):
            initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
            optimized_params = optimize_params(sample_X, initial_params)
            score = objective_function(optimized_params, sample_X)
            anomaly_scores_qead.append(score)

        smart_threshold_qead = calculate_smart_threshold(anomaly_scores_qead)
        y_pred_qead = [1 if score > smart_threshold_qead else 0 for score in anomaly_scores_qead]
        qead_metrics = calculate_metrics(y_test[:len(y_pred_qead)], y_pred_qead)
        nab_score_qead = calculate_nab_score(qead_metrics['TP Rate'], qead_metrics['TN Rate'], nab_weights)

        # Quantum Self-Attention (QSA)
        print(f"Running Quantum Self-Attention for dataset {name}...")
        anomaly_scores_qsa = []
        qsa = QuantumSelfAttention(window_size)
        for sample_X in tqdm(X_test, desc="QSA"):
            result = qsa.run(sample_X)
            score = qsa.process_attention_output(result)
            anomaly_scores_qsa.append(score)

        smart_threshold_qsa = calculate_smart_threshold(anomaly_scores_qsa)
        y_pred_qsa = [1 if score > smart_threshold_qsa else 0 for score in anomaly_scores_qsa]
        qsa_metrics = calculate_metrics(y_test[:len(y_pred_qsa)], y_pred_qsa)
        nab_score_qsa = calculate_nab_score(qsa_metrics['TP Rate'], qsa_metrics['TN Rate'], nab_weights)

        # DNN-based anomaly detection
        print(f"Running DNN Self-Attention for dataset {name}...")
        dnn_metrics = dnn_anomaly_detection(X_train, y_train, X_test, y_test, device)
        nab_score_dnn = calculate_nab_score(dnn_metrics['TP Rate'], dnn_metrics['TN Rate'], nab_weights)

        results[name] = {
            'qead_metrics': qead_metrics,
            'qsa_metrics': qsa_metrics,
            'dnn_metrics': dnn_metrics,
            'nab_scores': {
                'QEAD': nab_score_qead,
                'QSA': nab_score_qsa,
                'DNN': nab_score_dnn
            }
        }

    return results

# Print comparison table with NAB Score
def print_comparison_table(results):
    for dataset_name, res in results.items():
        print(f"\nComparison Table for Dataset: {dataset_name}")
        for method, metrics in res.items():
            if method == 'nab_scores':
                print(f"\nNAB Scores:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
            else:
                print(f"{method:<20} "
                      f"Accuracy: {metrics['Accuracy']:<8.3f} "
                      f"MCC: {metrics['MCC']:<8.3f} "
                      f"F1: {metrics['F1']:<8.3f} "
                      f"TP Rate: {metrics['TP Rate']:<8.3f} "
                      f"TN Rate: {metrics['TN Rate']:<8.3f}")

# Main script execution
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = load_datasets()
    results = run_comparison(datasets, window_size=4, device=device)
    print_comparison_table(results)

if __name__ == "__main__":
    main()
