import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import custom classes and functions
from Models.LSTM_Small import LSTMSmallNetwork
from Pytorch_Dataset_Training.LSTM_Small_Dataset import TimeSeriesDataset
from file_Paths.file_Paths import File_Path_Retriever

# Create an instance of the File_Path_Retriever class
filepath_ret = File_Path_Retriever()

# Retrieve the path to the numpy dataset
numpy_dataset_path = filepath_ret.return_numpy_file_path()

# Load the numpy dataset
loaded_data = np.load(numpy_dataset_path)

# Extract mean and std values over time
mean_values = loaded_data[:, 300, 300, 0]  # Mean values of the first pixel
std_values = loaded_data[:, 300, 300, 1]   # Standard deviation values of the first pixel

# Structure the data
X = []
Y = []

for i in range(0, 356):  # Adjust the range as needed
    mean_sequence = mean_values[i:i+10]
    std_sequence = std_values[i:i+10]
    X.append(np.column_stack((mean_sequence[:-1], std_sequence[:-1])))  # Input sequence with two features
    Y.append(mean_sequence[-1])   # Corresponding output value

# Train-test split
X = np.array(X)
Y = np.array(Y)

x_train, x_test = X[:300], X[300:]
y_train, y_test = Y[:300], Y[300:]

# Model hyperparameters
input_size = 1
hidden_size = 30
num_layers = 5
bidirectional = True

# Create the dataset and dataloader
dataset = TimeSeriesDataset(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)

# Model and optimizer initialization
input_size = 2  # Change input size to 2 for two features
model = LSTMSmallNetwork(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
epochs = 15

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

# Plot the results
import matplotlib.pyplot as plt

plt.plot(train_pred.numpy(), label='predicted')
plt.plot(y_train, label='original')
plt.legend()
plt.title('Training Set: Original vs Predicted')
plt.xlabel('Time')
plt.ylabel('Mean Value')
plt.show()

# Plot the results for the test set
plt.subplot(1, 2, 2)
plt.plot(test_pred.numpy(), label='predicted')
plt.plot(y_test, label='original')
plt.legend()
plt.title('Test Set: Original vs Predicted')
plt.xlabel('Time')
plt.ylabel('Mean Value')

plt.tight_layout()
plt.show()

