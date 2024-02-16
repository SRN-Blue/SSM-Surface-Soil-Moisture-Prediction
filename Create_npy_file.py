import os
import rasterio
import numpy as np
import re

# directory containing the SSM maps
data_dir = r'C:\Users\ASUS\Desktop\Surface Soil Moisture_Sentinel1_Fanap_Task_code\Dataset\S1_SSM'
npy_dir = r'C:\Users\ASUS\Desktop\Surface Soil Moisture_Sentinel1_Fanap_Task_code\Dataset\npy_files'

class Create_Npy_File():
    
    def __init__(self, dataset_path, save_path) -> None:
        self.dataset_path = dataset_path
        self.save_path = save_path

    def extract_dates(self) -> list:
        # Define the pattern for extracting dates from filenames
        date_pattern = re.compile(r'SSM_P_STCD_(\d{8})_.*\.img_mean')

        # List to store extracted dates
        extracted_dates = []

        # Iterate through filenames and extract dates
        for filename in os.listdir(self.dataset_path):
            match = date_pattern.match(filename)
            if match:
                extracted_dates.append(match.group(1))

        # Sort the dates in ascending order
        sorted_dates = sorted(extracted_dates)

        return sorted_dates
    
    def npy_combine_mead_std_one_file(self, date) -> tuple:
        mean_date_pattern = re.compile(r'SSM_P_STCD_(\d{8})_.*\.img_mean$')
        stddev_date_pattern = re.compile(r'SSM_P_STCD_(\d{8})_.*\.img_stddev$')

        # Iterate through filenames and find the file with the specific date
        for mean_filename in os.listdir(self.dataset_path):
            mean_match = mean_date_pattern.match(mean_filename)
            # if found the mean for that date
            if mean_match and mean_match.group(1) == date:
                for stdn_filename in os.listdir(self.dataset_path):
                    stddev_match = stddev_date_pattern.match(stdn_filename)
                    # if also found std file for that date
                    if stddev_match and stddev_match.group(1) == date:
                        mean_data_file = os.path.join(self.dataset_path, mean_filename)
                        stdn_data_file = os.path.join(self.dataset_path, stdn_filename)
                        # Open the data files
                        with rasterio.open(mean_data_file) as mean_src, rasterio.open(stdn_data_file) as stddev_src:
                            # Read the data
                            mean_ssm_data = mean_src.read(1)  # it's a single-band raster
                            stddev_ssm_data = stddev_src.read(1)  # it's a single-band raster

                            # # Combine mean and stddev data into a single 3D array
                            # combined_data = np.stack((mean_ssm_data, stddev_ssm_data), axis=-1)
                            return mean_ssm_data, stddev_ssm_data
                    # else:
                    #     print(f"std Data files for date {date} not found(while there was mean file).")
            # else:
            #     print(f"Mean Data files for date {date} not found.")
        return False
    
    def stack_mean_std_temporal(self) -> list:
        dates = self.extract_dates()
        print(f'dates are: {dates}')
        print(f'len of dates are: {len(dates)}')

        # List to accumulate individual results
        combined_data_list = []

        for date in dates:
            mean_ssm_data, stddev_ssm_data = self.npy_combine_mead_std_one_file(date)

            # Normalize mean and std values
            mean_ssm_data_normalized = (mean_ssm_data / 10000.0 - 0.015) / 0.585
            stddev_ssm_data_normalized = stddev_ssm_data / 10000.0 / 0.585

            # combined_data_list.append(np.stack((mean_ssm_data, stddev_ssm_data), axis=-1))
            combined_data_list.append(np.stack((mean_ssm_data_normalized, stddev_ssm_data_normalized), axis=-1))

        # Stack the individual results into one 4D matrix
        final_combined_data = np.stack(combined_data_list, axis=0)
        # Now final_combined_data is a 4D matrix containing normalized data for different dates
        print(final_combined_data.shape)

        return final_combined_data

    def save_stacked_npy_file(self, npy_stacked) -> None:
        np.save(self.save_path, npy_stacked)
        print(f'sucessfully saved npy in path: {self.save_path}.\n Enjoy :)')


createnpy = Create_Npy_File(data_dir, npy_dir)
stacked_npy = createnpy.stack_mean_std_temporal()
createnpy.save_stacked_npy_file(stacked_npy)