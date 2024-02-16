import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import custom classes and functions
from Models.LSTM_Small import LSTMSmallNetwork
from Pytorch_Dataset_Training.LSTM_Small_Dataset import TimeSeriesDataset
from file_Paths.file_Paths import File_Path_Retriever

# Initialize file path retriever
filepath_ret = File_Path_Retriever()

# Load numpy data
loaded_data = np.load(filepath_ret)

# Assuming the shape of data is (num_samples, height, width, num_features)
num_samples, height, width, num_features = loaded_data.shape

# Extract mean and std values over time for all pixels
mean_values = loaded_data[:, :, :, 0].reshape(num_samples, -1)  # Flatten along height and width
std_values = loaded_data[:, :, :, 1].reshape(num_samples, -1)

# Structure the data
X = []
Y = []

sequence_length = 10  # Assuming a sequence length of 10

for i in range(0, num_samples - sequence_length + 1):  # Adjust the range as needed
    mean_sequence = mean_values[i:i+sequence_length].flatten()
    std_sequence = std_values[i:i+sequence_length].flatten()
    X.append(np.column_stack((mean_sequence[:-1], std_sequence[:-1])))  # Input sequence with two features
    Y.append(mean_sequence[-1])   # Corresponding output value

# Train-test split
X = np.array(X)
Y = np.array(Y)

# Assuming a split ratio of 80-20
split_ratio = 0.8
split_idx = int(split_ratio * len(X))

x_train, x_test = X[:split_idx], X[split_idx:]
y_train, y_test = Y[:split_idx], Y[split_idx:]

# Model hyperparameters
input_size = 2  # Change input size to 2 for two features
hidden_size = 10
num_layers = 2
bidirectional = True

# Create the dataset and dataloader
dataset = TimeSeriesDataset(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)


input_size = 1
hidden_size = 20
num_layers = 5
bidirectional = True

# Model and optimizer initialization
input_size = 2  # Change input size to 2 for two features
model = LSTMSmallNetwork(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 1500

# Training loop
for epoch in range(epochs):
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        input_sequences = data[0].view(-1, 9, input_size)
        target_values = data[1]

        y_pred = model(input_sequences).reshape(-1)
        loss = criterion(y_pred, target_values)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

# After training, check the performance on the training set
model.eval()
with torch.no_grad():
    train_pred = model(torch.tensor(x_train, dtype=torch.float32).view(-1, 9, input_size)).reshape(-1)
    test_pred = model(torch.tensor(x_test, dtype=torch.float32).view(-1, 9, input_size)).reshape(-1)


