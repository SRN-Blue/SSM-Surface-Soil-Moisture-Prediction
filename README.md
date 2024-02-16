# SSM Forecasting Project

## Dataset

This project focuses on forecasting Surface Soil Moisture (SSM) using the "Dataset of Sentinel-1 surface soil moisture time series at 1 km resolution over Southern Italy" [Balenzano et al., 2021](https://doi.org/10.5281/zenodo.5006307). The primary goal is to develop models for predicting SSM based on Sentinel-1 satellite data.

### Dataset Information

- **Name**: Dataset of Sentinel-1 surface soil moisture time series at 1 km resolution over Southern Italy
- **Citation**: Balenzano, A., Mattia, F., Satalino, G., Lovergine, F. P., & Palmisano, D. (2021). Dataset of Sentinel-1 surface soil moisture time series at 1 km resolution over Southern Italy (Version 3) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.5006307](https://doi.org/10.5281/zenodo.5006307)

## Data Preprocessing

The initial step involves converting the dataset files into a 4D NumPy file. This conversion is performed using the `Create_npy_file.py` script found in the `Data_Preprocess` module.

## Data Visualization

The `Visualization_data` module provides a set of visualization tools to better understand the dataset. Key visualization functions include:

- `report_each_axes_summary()`: Summarizes information for each axis in the dataset.
- `hdr_file_plot()`: Generates plots based on HDR files in the dataset.
- `plot_ssm_mean_over_time()`: Visualizes the mean surface soil moisture over time.

## Models

The project employs three different models for training:

1. **Autoformer**
2. **ConvLSTM**
3. **LSTM_Small**

Due to limitations in available RAM, the training of the `LSTM_Small` model has been completed.

## Training Files

There are three distinct training scripts, each serving a specific purpose:

1. **train_allsamples_all_features.py**: Considers all samples (pixels over time with standard deviation and mean). Note that this script has not been tested due to RAM limitations.
2. **train_Single_shot_one_feature.py**: Trains the model using one pixel over time with a single feature (Mean).
3. **train_Single_shot_two_feature.py**: Trains the model using one pixel over time with two features (Mean and Standard Deviation).

These training scripts provide flexibility in choosing the type of training data and feature combinations.

## Code Explanations

# LSTMSmallNetwork Model Documentation

## Overview

This documentation explains the `LSTMSmallNetwork` model, a small-scale Long Short-Term Memory (LSTM) neural network implemented using PyTorch. The model is designed for sequence-to-one regression tasks, particularly suited for time series forecasting.

## Model Architecture

### LSTM Layer

- **Input Size**: Specifies the number of features in the input sequence (default: 1).
- **Hidden Size**: Defines the number of features in the hidden state of the LSTM (default: 10).
- **Number of Layers**: Specifies the number of LSTM layers (default: 2).
- **Bidirectional**: Determines whether the LSTM layers are bidirectional (default: True).

### Fully Connected Layer

- **Input Size for FC Layer**: Adjusted based on bidirectionality. If bidirectional, the input size is twice the hidden size; otherwise, it's equal to the hidden size.
- **Output Size**: Fixed to 1, suitable for sequence-to-one regression.

## Forward Method

The forward method processes input sequences through the LSTM layers and extracts the output from the last time step. This output is then fed into a fully connected (linear) layer with ReLU activation.

```python
def forward(self, x):
    output, _status = self.lstm(x)
    output = output[:, -1, :]  # Take the output from the last time step
    output = self.fc1(torch.relu(output))
    return output

