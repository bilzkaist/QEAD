import torch
import torch.nn as nn

# Define various DNN models: CNN, LSTM, GRU, CNN-LSTM, CNN-GRU, CNN-MHA, and Self-Attention DNN
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Ensure the input has the right shape (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        return self.fc1(lstm_out[:, -1, :])  # Extract the last time-step output

# Define GRU Model in PyTorch
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Ensure the input has the right shape (batch, seq_len, input_size)
        gru_out, _ = self.gru(x)
        return self.fc1(gru_out[:, -1, :])  # Extract the last time-step output

# Define CNN-LSTM Model in PyTorch
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(16, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Ensure the input has the right shape (batch, channels, input_size)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # LSTM expects (batch, seq_len, input_size), adjust dimensions
        x, _ = self.lstm(x)
        return self.fc1(x[:, -1, :])  # Extract the last time-step output

class CNNGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)  # 16 output channels
        self.gru = nn.GRU(16, hidden_size, batch_first=True)      # GRU expects 16 input features
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for CNN (batch_size, 1, input_size)
        x = self.relu(self.conv1(x))  # (batch_size, 16, input_size)
        x = x.permute(0, 2, 1)  # Change to (batch_size, input_size, 16) for GRU
        x, _ = self.gru(x)  # (batch_size, input_size, hidden_size)
        return self.fc1(x[:, -1, :])  # Extract the last time-step output

class CNNMHA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNMHA, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.mha = nn.MultiheadAttention(embed_dim=16, num_heads=4)
        self.fc1 = nn.Linear(16, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        x = self.relu(self.conv1(x))
        x = x.permute(2, 0, 1)  # MultiheadAttention expects (seq_len, batch, embed_dim)
        attn_output, _ = self.mha(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # Convert back to (batch, seq_len, embed_dim)
        x = self.relu(self.fc1(attn_output[:, -1, :]))
        return self.fc2(x)
    
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
