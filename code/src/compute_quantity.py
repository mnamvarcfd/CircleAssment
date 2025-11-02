import numpy as np
from logging_config import setup_logger

logger = setup_logger()

class ComputeQuantity:
    """
    A class for computing quantities from DICOM data and masks
    """
    def __init__(self, frames: np.ndarray, blood_pool_mask: np.ndarray, myo_mask: np.ndarray):
        self.frames = frames
        self.blood_pool_mask = blood_pool_mask
        self.myo_mask = myo_mask


    def _extract_pixel_time_series(self, mask: np.ndarray)->tuple[list, np.ndarray]:
        """
        Extract time series data for each pixel within the mask

        Args:
            mask (numpy.ndarray): Binary mask for the region of interest

        Returns:
            tuple: (pixel_coordinates, pixel_time_series)
                - pixel_coordinates: list of (y, x) coordinates where mask == 1
                - pixel_time_series: 2D array with shape (num_pixels, num_frames)
        """
        logger.info("Extracting time series data for each pixel within the mask")

        # Get coordinates where mask == 1 (binary mask)
        y_coords, x_coords = np.where(mask == 1)

        # Create a list of (y, x) coordinates where mask == 1
        pixel_coordinates = list(zip(y_coords, x_coords))

        # Get the number of pixels
        num_pixels = len(pixel_coordinates)

        # Get the number of frames
        num_frames = self.frames.shape[0]

        logger.info(f"Found {num_pixels} pixels within the mask")

        # Initialize array to store time series for each pixel
        pixel_time_series = np.zeros((num_pixels, num_frames))

        # Extract time series for each pixel
        for i, (y, x) in enumerate(pixel_coordinates):
            for t in range(num_frames):
                pixel_time_series[i, t] = self.frames[t, y, x]

        logger.info(f"Extracted time series data with shape {pixel_time_series.shape}")
        return pixel_coordinates, pixel_time_series


    def myocardium_time_series(self)->tuple[list, np.ndarray]:
        """
        Extract time series data for each pixel within the myocardium region

        Returns:
            tuple[list, np.ndarray]: _description_
        """
        logger.info("Extracting time series data for each pixel within the myocardium region")
        
        MYO_pixel_coordinates, MYO_time_series = self._extract_pixel_time_series(self.myo_mask)
        
        return MYO_pixel_coordinates, MYO_time_series
    

    def blood_pool_time_series(self)->tuple[list, np.ndarray]:
        """
        Extract time series data for each pixel within the blood pool region

        Returns:
            tuple[list, np.ndarray]: (pixel_coordinates, pixel_time_series)
        """
        logger.info("Extracting time series data for each pixel within the blood pool region")
        
        blood_pool_pixel_coordinates, blood_pool_time_series = self._extract_pixel_time_series(self.blood_pool_mask)
        
        return blood_pool_pixel_coordinates, blood_pool_time_series
    
     
    def arterial_input_function(self)->np.ndarray:
        """
        Compute the Arterial Input Function (AIF) by averaging pixel values across all pixels at each time stamp

        This method computes the AIF(t) by averaging the signal intensity across all pixels within
        the blood pool for each time stamp. For each time stamp, it takes the mean of all pixel
        values at that time, resulting in a single average intensity value per time frame.

        Returns:
            numpy.ndarray: 1D array of AIF values with shape (num_frames,)
                - Each element represents the average signal intensity across all pixels
                  within the blood pool at that specific time stamp
                - Length equals the number of time stamps (num_time_stamps)
        """
        logger.info("Computing AIF(t)")

        # Extract time series for each pixel within the blood pool region
        pixel_coordinates, pixel_time_series = self._extract_pixel_time_series(self.blood_pool_mask)

        num_time_stamps = pixel_time_series.shape[1]
        aif = np.zeros(num_time_stamps)

        # Compute average signal intensity of the blood pool pixels
        aif = np.mean(pixel_time_series, axis=0)

        logger.info(f"AIF computed with {num_time_stamps} time stamp")
        return aif
   
        
        
if __name__ == "__main__":

    from data_loader import DataLoader
    data_loader = DataLoader(dicom_dir="input_data/DICOM_files")
    frames = data_loader.dicom()
    aif_mask = data_loader.mask(mask_index=0)  # Blood pool
    myo_mask = data_loader.mask(mask_index=1)  # Myocardium
    compute_quantity = ComputeQuantity(frames=frames, aif_mask=aif_mask, myo_mask=myo_mask)
    
    aif =compute_quantity.arterial_input_function()

    from save_data_manager import SaveDataManager
    save_data_manager = SaveDataManager()
    save_data_manager.plot_pixel_over_time(aif,
                                 title=f"arterial_input_function",
                                 y_label="Signal Intensity",
                                 output_filename="arterial_input_function.png")
    
    myo_pixel_coordinates, myo_time_series = compute_quantity.myocardium_time_series()
    save_data_manager.plot_pixel_over_time(myo_time_series[10],
                                 title=f"myocardium_time_series_10",
                                 y_label="Signal Intensity",
                                 output_filename="myocardium_time_series_10.png")
    