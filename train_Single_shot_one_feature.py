import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import custom classes and functions
from Models.LSTM_Small import LSTMSmallNetwork
from Pytorch_Dataset_Training.LSTM_Small_Dataset import TimeSeriesDataset
from file_Paths.file_Paths import File_Path_Retriever

# Initialize file path retriever
filepath_ret = File_Path_Retriever()

# Load numpy data
loaded_data = np.load(filepath_ret)

# Extract mean values over time
mean_values = loaded_data[:, 300, 300, 0]  # Assuming we want the mean value of the 300, 300 pixel

# Structure the data
X = []
Y = []

for i in range(0, 356):  # All temporal data
    sequence = mean_values[i:i+10]
    X.append(sequence[:-1])  # Input sequence
    Y.append(sequence[-1])   # Corresponding output value

# Train-test split
X = np.array(X)
Y = np.array(Y)

x_train, x_test = X[:300], X[300:]
y_train, y_test = Y[:300], Y[300:]

# Create the dataset and dataloader
dataset = TimeSeriesDataset(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)

# Model hyperparameters
input_size = 1
hidden_size = 10
num_layers = 2
bidirectional = True

# Initialize the model
model = LSTMSmallNetwork(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)

# Initialize optimizer and loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
epochs = 15000

# Training loop
for epoch in range(epochs):
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        input_sequences = data[0].view(-1, 9, 1)
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
    train_pred = model(torch.tensor(x_train, dtype=torch.float32).view(-1, 9, 1)).reshape(-1)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(train_pred.numpy(), label='predicted')
plt.plot(y_train, label='original')
plt.legend()
plt.show()

