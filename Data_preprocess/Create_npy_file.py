import os
import rasterio
import numpy as np
import re

from file_Paths.file_Paths import File_Path_Retriever

# Initialize file path retriever
filepath_ret = File_Path_Retriever()


class Create_Npy_File():
    
    def __init__(self, dataset_path, save_path) -> None:
        """
        Initialize the CreateNpyFile class.

        Parameters:
        - dataset_path (str): Path to the directory containing SSM maps.
        - save_path (str): Path to save the final combined numpy file.
        """
        self.dataset_path = dataset_path
        self.save_path = save_path

    def extract_dates(self) -> list:
        """
        Extract and sort dates from filenames.

        Returns:
        - list: Sorted list of extracted dates.
        """
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
        """
        Combine mean and standard deviation data for a specific date.

        Parameters:
        - date (str): Specific date.

        Returns:
        - tuple: (mean_ssm_data, stddev_ssm_data) if found, False otherwise.
        """
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
        """
        Stack mean and standard deviation data for different dates.

        Returns:
        - np.ndarray: Stacked 4D matrix containing normalized data for different dates.
        """
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
        """
        Save the stacked numpy file.

        Parameters:
        - npy_stacked (np.ndarray): Stacked 4D matrix containing normalized data for different dates.
        """
        np.save(self.save_path, npy_stacked)
        print(f'sucessfully saved npy in path: {self.save_path}.\n Enjoy :)')


# directory containing the SSM maps
data_dir = filepath_ret.return_data_dir_path
npy_dir = filepath_ret.return_npy_save_dir_path


# Create instance of CreateNpyFile
createnpy = Create_Npy_File(data_dir, npy_dir)

# Stack mean and standard deviation data for different dates
stacked_npy = createnpy.stack_mean_std_temporal()

# Save the stacked numpy file
createnpy.save_stacked_npy_file(stacked_npy)



