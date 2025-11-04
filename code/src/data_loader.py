import os
import numpy as np
import pandas as pd
import pydicom
import tifffile
from logging_config import setup_logger

# Setup logger using centralized configuration
logger = setup_logger()


class DataLoader:
    """
    A class for loading DICOM data and masks from a specified input directory.
    
    This class encapsulates data loading functionality and stores the input data
    directory path, making it easier to manage and reuse data loading operations.
    """
    
    def __init__(self, dicom_dir: str = "input_data/DICOM_files", mask_file: str = "input_data/AIF_And_Myo_Masks.tiff"):
        """
        Initialize the DataLoader with input data directories.
        
        Args:
            dicom_dir (str): Directory containing the DICOM files. 
                           Defaults to "input_data/DICOM_files" (relative to project root).
            mask_file (str): Path to the TIFF file containing masks.
                           Defaults to "input_data/AIF_And_Myo_Masks.tiff" (relative to project root).
        """
        # Resolve paths relative to project root if they are relative paths
        # This ensures paths work regardless of where the script is run from
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up from src/ to project root
        
        # Resolve DICOM directory path
        if os.path.isabs(dicom_dir):
            self.dicom_dir = dicom_dir
        else:
            # Resolve relative to project root
            self.dicom_dir = os.path.normpath(os.path.join(project_root, dicom_dir))
        
        # Resolve mask file path
        if os.path.isabs(mask_file):
            self.mask_file = mask_file
        else:
            # Resolve relative to project root
            self.mask_file = os.path.normpath(os.path.join(project_root, mask_file))
        
        logger.info(f"DICOM directory: {self.dicom_dir}")
        logger.info(f"Mask file: {self.mask_file}")
        
        # Validate that directories/files exist
        if not os.path.isdir(self.dicom_dir):
            logger.warning(f"DICOM directory not found: {self.dicom_dir}")
        
        if not os.path.isfile(self.mask_file):
            logger.warning(f"Mask file not found: {self.mask_file}")
    
        # Initialize time array (will be populated when dicom() is called)
        self.time = np.array([], dtype=np.float64)
        
    def dicom(self) -> np.ndarray:
        """
        Load a series of DICOM files and return as a 3D array: frames[t,y,x]
        
        Also extracts Acquisition Time tags from each DICOM file and stores elapsed time
        (relative to first file) in self.time as a numpy array.
        - First file's elapsed time = 0.0
        - Subsequent files show time difference from first file in seconds
        
        Returns:
            numpy.ndarray: 3D array of DICOM frames with shape (num_frames, height, width)
        """
        logger.info(f"Loading DICOM files from {self.dicom_dir}")
        
        # Check if directory exists before trying to list it
        if not os.path.isdir(self.dicom_dir):
            raise FileNotFoundError(
                f"DICOM directory not found: {self.dicom_dir}\n"
                f"Absolute path: {os.path.abspath(self.dicom_dir)}\n"
                f"Current working directory: {os.getcwd()}"
            )
        
        # List all DICOM files
        dicom_files = [f for f in sorted(os.listdir(self.dicom_dir)) if f.endswith('.dcm')]
        
        if len(dicom_files) == 0:
            raise ValueError(f"No DICOM files found in {self.dicom_dir}")
        
        # Read the first file to get dimensions
        first_dicom = pydicom.dcmread(os.path.join(self.dicom_dir, dicom_files[0]))
        img_shape = first_dicom.pixel_array.shape
        
        # Create 3D array to hold all frames
        frames = np.zeros((len(dicom_files), img_shape[0], img_shape[1]), 
                         dtype=first_dicom.pixel_array.dtype)
        
        # Array to store acquisition times (in seconds)
        time_values = []
        
        # Load each DICOM file and extract acquisition time
        for i, filename in enumerate(dicom_files):
            dicom_data = pydicom.dcmread(os.path.join(self.dicom_dir, filename))
            frames[i] = dicom_data.pixel_array
            
            # Extract Acquisition Time tag (0008,0032)
            acquisition_time = None
            if hasattr(dicom_data, 'AcquisitionTime'):
                acquisition_time = dicom_data.AcquisitionTime
            elif (0x0008, 0x0032) in dicom_data:
                acquisition_time = dicom_data[(0x0008, 0x0032)].value
            
            # Parse acquisition time to seconds
            if acquisition_time is not None:
                time_seconds = self._parse_dicom_time_to_seconds(str(acquisition_time))
                time_values.append(time_seconds)
            else:
                time_values.append(None)
        
        # Calculate elapsed time relative to first file
        if len(time_values) > 0 and time_values[0] is not None:
            first_time = time_values[0]
            elapsed_times = []
            for i, time_sec in enumerate(time_values):
                if i == 0:
                    # First file is always 0
                    elapsed_times.append(0.0)
                elif time_sec is not None:
                    # Calculate difference from first file
                    elapsed_time = time_sec - first_time
                    elapsed_times.append(elapsed_time)
                else:
                    # No valid time for this file
                    elapsed_times.append(np.nan)
            
            # Store elapsed times in self.time
            self.time = np.array(elapsed_times, dtype=np.float64)
        else:
            logger.warning("No valid acquisition times found in DICOM files. Setting self.time to empty array.")
            self.time = np.array([], dtype=np.float64)
        
        logger.info(f"Successfully loaded {len(dicom_files)} DICOM files from {self.dicom_dir} "
                   f"with shape {frames.shape}")
        logger.info(f"Extracted {len(self.time)} time values (elapsed time in seconds)")
        return frames
    
    def mask(self, mask_index: int = 0) -> np.ndarray:
        """
        Load the mask from the TIFF file
        
        Args:
            mask_index (int): Index of the mask to load (0 for blood pool, 1 for myocardium, etc.)
                            Defaults to 0.
        
        Returns:
            numpy.ndarray: Binary mask for the selected region
        """
        logger.info(f"Loading mask (index {mask_index}) from {self.mask_file}")
        
        # Check if file exists before trying to read it
        if not os.path.isfile(self.mask_file):
            raise FileNotFoundError(
                f"Mask file not found: {self.mask_file}\n"
                f"Absolute path: {os.path.abspath(self.mask_file)}\n"
                f"Current working directory: {os.getcwd()}"
            )
        
        # Load the specified page of the TIFF file
        mask = tifffile.imread(self.mask_file, key=mask_index)
        
        # Ensure mask is binary (0 and 1)
        mask = (mask > 0).astype(np.uint8)
        
        logger.info(f"Mask loaded with shape {mask.shape}")
        return mask
    
    def _parse_dicom_time_to_seconds(self, time_str: str) -> float:
        """
        Parse DICOM time string (HHMMSS.FFFFFF) to total seconds.
        
        Args:
            time_str: DICOM time string in format HHMMSS or HHMMSS.FFFFFF
        
        Returns:
            float: Total seconds since midnight, or None if parsing fails
        """
        if time_str is None or time_str == 'N/A' or time_str == 'ERROR':
            return None
        
        try:
            time_str = str(time_str).strip()
            
            # Handle format: HHMMSS.FFFFFF or HHMMSS
            if '.' in time_str:
                time_part, fraction_part = time_str.split('.')
            else:
                time_part = time_str
                fraction_part = '0'
            
            # Ensure time_part is at least 6 digits (HHMMSS)
            if len(time_part) < 6:
                logger.warning(f"Time string too short: {time_str}")
                return None
            
            # Extract hours, minutes, seconds
            hours = int(time_part[0:2])
            minutes = int(time_part[2:4])
            seconds = int(time_part[4:6])
            
            # Parse fractional seconds
            fraction = float('0.' + fraction_part) if fraction_part else 0.0
            
            # Calculate total seconds
            total_seconds = hours * 3600 + minutes * 60 + seconds + fraction
            
            return total_seconds
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing time string '{time_str}': {str(e)}")
            return None
    
    
if __name__ == "__main__":
    
    data_loader = DataLoader(dicom_dir="input_data/DICOM_files")
