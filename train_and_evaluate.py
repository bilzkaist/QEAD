import torch
import time
import psutil
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, matthews_corrcoef

def get_model_stats(model, input_size, device):
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input_sample = torch.randn(1, input_size).to(device)
    model = model.to(device)

    with torch.no_grad():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        _ = model(input_sample)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        single_pred_time = time.time() - start_time

    memory_usage = psutil.virtual_memory().used / (1024 ** 2)
    return model_parameters, single_pred_time, memory_usage

def train_dnn_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    tp_rate = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'Accuracy': accuracy,
        'F1': f1,
        'MCC': mcc,
        'TP Rate': tp_rate,
        'TN Rate': tn_rate
    }

def evaluate_and_save_results(model, test_loader, device, model_type, X_train, results_file, training_time):
    y_pred, y_true = [], []
    model_parameters, single_pred_time, memory_usage = get_model_stats(model, X_train.shape[1], device)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int)
            y_pred.extend(predictions.flatten())
            y_true.extend(labels.numpy())

    metrics = calculate_metrics(y_true, y_pred)

    with open(results_file, 'a') as f:
        f.write(f"\nModel: {model_type}\nMetrics: {metrics}\n")
    
    return metrics
