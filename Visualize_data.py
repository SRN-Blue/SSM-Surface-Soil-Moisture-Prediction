# import os
# import rasterio
# from rasterio.plot import show
# import cv2 as cv
# from matplotlib import pyplot as plt

# # Example directory containing the SSM maps
# data_dir = r'C:\Users\ASUS\Desktop\Surface Soil Moisture_Sentinel1_Fanap_Task_code\Dataset\S1_SSM'

# # List all files in the directory
# file_list = os.listdir(data_dir)

# # Sort the files by date
# file_list.sort()

# # Visualize a few SSM maps
# num_maps_to_visualize = 3

# for i in range(num_maps_to_visualize):
#     file_path = os.path.join(data_dir, file_list[i])

#     with rasterio.open(file_path) as src:
#         ssm_map = src.read(1)  # Assuming it's a single-band raster
#         vmin, vmax = 0, 0.6  # Adjust based on your data range

#         plt.figure(figsize=(8, 8))
#         show(ssm_map, cmap='jet', vmin=vmin, vmax=vmax, title=f'SSM Map - {file_list[i]}')


import os
import rasterio
from rasterio.plot import show
import cv2 as cv
from matplotlib import pyplot as plt

# Example directory containing the SSM maps
data_dir = r'C:\Users\ASUS\Desktop\Surface Soil Moisture_Sentinel1_Fanap_Task_code\Dataset\S1_SSM'

# List all files in the directory
file_list = os.listdir(data_dir)

# Sort the files by date
file_list.sort()

# Visualize a few SSM maps
num_maps_to_visualize = 3

for i in range(num_maps_to_visualize):
    # Identify the data file associated with the header file
    hdr_file = os.path.join(data_dir, file_list[i] + '.hdr')
    data_file = hdr_file.replace('.hdr', '')

    # Open the data file instead of the header file
    with rasterio.open(data_file) as src:
        ssm_map = src.read(1)  # Assuming it's a single-band raster
        vmin, vmax = 0, 0.6  # Adjust based on your data range

        plt.figure(figsize=(8, 8))
        show(ssm_map, cmap='jet', vmin=vmin, vmax=vmax, title=f'SSM Map - {file_list[i]}')



