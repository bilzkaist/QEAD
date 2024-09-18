from datasets import load_datasets, preprocess_data
from quantum_methods import optimize_params, QuantumSelfAttention
from dnn_models import CNNModel, LSTMModel, GRUModel,CNNLSTMModel, CNNGRUModel, CNNMHA
from train_and_evaluate import train_dnn_model, evaluate_and_save_results
from classical_methods import classical_methods

import torch

import warnings
warnings.filterwarnings("ignore")

# Set dataset path and file name for the Bearing dataset
dataset_path = "/home/bilz/datasets/qead/q/"
dataset_name = "Bearing.csv"
dataset_filename = dataset_path + dataset_name 
results_path = "/home/bilz/results/" 
qubit_number = 4


# Main script execution
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = load_datasets(dataset_filename)
    results, y_test_dict = run_comparison(datasets, qubit_no=qubit_number, device=device)
    print_comparison_table(results, y_test_dict)

if __name__ == "__main__":
    main()

