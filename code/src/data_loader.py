import os
import numpy as np
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
    
    def dicom(self) -> np.ndarray:
        """
        Load a series of DICOM files and return as a 3D array: frames[t,y,x]
        
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
        
        # Load each DICOM file
        for i, filename in enumerate(dicom_files):
            dicom_data = pydicom.dcmread(os.path.join(self.dicom_dir, filename))
            frames[i] = dicom_data.pixel_array
        
        logger.info(f"Successfully loaded {len(dicom_files)} DICOM files from {self.dicom_dir} "
                   f"with shape {frames.shape}")
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

