import os
import rasterio
from rasterio.plot import show
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# directory containing the SSM maps
data_dir = r'C:\Users\ASUS\Desktop\Surface Soil Moisture_Sentinel1_Fanap_Task_code\Dataset\S1_SSM'
npy_path = r'C:\Users\ASUS\Desktop\Surface Soil Moisture_Sentinel1_Fanap_Task_code\Dataset\npy_files\npy_files.npy'

class Visualization():
    def __init__(self, data_dir, npy_path) -> None:
        self.data_dir = data_dir
        self.npy_path = npy_path

    def hdr_file_plot(self):
        # List all files in the directory
        file_list = os.listdir(self.data_dir)

        # Sort the files by date
        file_list.sort()

        # Visualize a few SSM maps
        num_maps_to_visualize = 3

        for i in range(num_maps_to_visualize):
            # Identify the data file associated with the header file
            hdr_file = os.path.join(self.data_dir, file_list[i] + '.hdr')
            data_file = hdr_file.replace('.hdr', '')

            # Open the data file instead of the header file
            with rasterio.open(data_file) as src:
                num_bands = src.count
                print(num_bands)
                ssm_map = src.read(1)
                vmin, vmax = 0, 0.6

                plt.figure(figsize=(8, 8))
                show(ssm_map, cmap='jet', vmin=vmin, vmax=vmax, title=f'SSM Map - {file_list[i]}')
    
    def plot_ssm_mean_over_time(self):
        # Load the saved Numpy array
        loaded_data = np.load(self.npy_path)

        # plot the time series for a pixel at position (x, y)
        x, y = 100, 100  # Replace these with the desired pixel coordinates

        # Extract time series data for the specified pixel
        pixel_time_series = loaded_data[:, x, y, 0]  # first channel represents mean data

        # Plot the time series
        plt.figure(figsize=(10, 6))
        plt.plot(pixel_time_series, label=f'Pixel ({x}, {y}) Mean Time Series')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Mean SSM Value')
        plt.title('Time Series of Surface Soil Moisture for a Specific Pixel')
        plt.legend()
        plt.show()

    def plot_ssm_mean_over_time_nonzero(self):
        # Load the saved Numpy array
        loaded_data = np.load(self.npy_path)

        # Iterate over pixels and find the first one with non-zero values throughout the time series
        non_zero_pixel_found = False
        for x in range(loaded_data.shape[1]):
            for y in range(loaded_data.shape[2]):
                pixel_time_series = loaded_data[:, x, y, 0]  # the first channel represents mean data
                if np.all(pixel_time_series != 0):
                    non_zero_pixel_found = True
                    break
            if non_zero_pixel_found:
                break

        # Plot the time series for the first pixel found with non-zero values
        plt.figure(figsize=(10, 6))
        plt.plot(pixel_time_series, label=f'Pixel ({x}, {y}) Mean Time Series')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Mean SSM Value')
        plt.title('Time Series of Surface Soil Moisture for a Specific Pixel')
        plt.legend()
        plt.show()

    def report_each_axes_summary(self):

        loaded_data = np.load(self.npy_path)
        # Print the shape of the loaded data
        print("Shape of loaded data:", loaded_data.shape)

        # Extract mean and std features
        mean_feature = loaded_data[:, :, :, 0]
        std_feature = loaded_data[:, :, :, 1]

        # Print information about the mean feature
        print("\nMean Feature:")
        print("Distinct values:", np.unique(mean_feature))
        print("Minimum value:", np.min(mean_feature))
        print("Maximum value:", np.max(mean_feature))

        # Print information about the std feature
        print("\nStandard Deviation Feature:")
        print("Distinct values:", np.unique(std_feature))
        print("Minimum value:", np.min(std_feature))
        print("Maximum value:", np.max(std_feature))


visulaizer = Visualization(data_dir, npy_path)
visulaizer.hdr_file_plot()
visulaizer.plot_ssm_mean_over_time_nonzero()




