import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless operation
import matplotlib.pyplot as plt

from logging_config import setup_logger


logger = setup_logger()


class SaveDataManager:
    """
    Manage saving operations for preloaded DICOM data (no disk reads here):
      - Save PNGs from preloaded frames
      - Write metadata JSONs from preloaded pydicom datasets
      - Create a movie from preloaded frames
    All outputs are written to the results directory.
    """

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = results_dir
        # Check if results directory exists, create it if it doesn't
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
            logger.info(f"Created results directory: {self.results_dir}")
        else:
            logger.debug(f"Results directory already exists: {self.results_dir}")


    def save_image(self, 
                   pixel_values: np.ndarray,
                   mask: np.ndarray,
                   Value_title: str,
                   output_filename: str) -> Optional[str]:
        """
        Save pixel values as a 2D image map using a mask for spatial reference.

        Args:
            pixel_values (np.ndarray): 1D array of values for each pixel (e.g., MBF values)
            mask (np.ndarray): 2D binary mask array for spatial reference
            output_filename (str): Name of the file to save the plot

        Returns:
            Optional[str]: Path to saved file if successful, None otherwise
        """
        pixel_values = np.asarray(pixel_values)
        
        mask = np.asarray(mask)

        mask_shape = mask.shape

        # Create output directory if it doesn't exist
        output_file = os.path.join(self.results_dir, output_filename)

        # Get pixel coordinates where mask == 1
        y_coords, x_coords = np.where(mask == 1)

        # Validate that the number of pixels matches
        if len(pixel_values) != len(y_coords):
            logger.error(f"save_image: pixel_values has {len(pixel_values)} pixels, but mask has {len(y_coords)} pixels")
            return None

        # Create a 2D array for visualization (filled with NaN for pixels outside mask)
        image_2d = np.full(mask_shape, np.nan, dtype=np.float64)

        # Fill in the pixel values at their corresponding pixel locations
        for i, (y, x) in enumerate(zip(y_coords, x_coords)):
            image_2d[y, x] = pixel_values[i]

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Plot the image map
        im = plt.imshow(image_2d, cmap='hot', interpolation='nearest')
        plt.title('Distribution of ' + Value_title)
        plt.xlabel('X pixel index')
        plt.ylabel('Y pixel index')

        # Add colorbar with label
        cbar = plt.colorbar(im, label=Value_title)

        # Add statistics text
        valid_values = pixel_values[pixel_values > 0]  # Exclude zeros from failed fits
        if len(valid_values) > 0:
            stats_text = f'Mean: {np.mean(valid_values):.2f}\n'
            stats_text += f'Max: {np.max(valid_values):.2f}\n'
            stats_text += f'Min: {np.min(valid_values):.2f}\n'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save the plot
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Image map saved to: {output_file}")
            plt.close()
            return output_file
        except Exception as e:
            logger.error(f"save_image: error saving image: {e}")
            plt.close()
            return None


    def create_movie_from_frames(self, 
                                 frames: np.ndarray, 
                                 output_path: str = "dicom_movie.avi", 
                                 fps: int = 10, 
                                 normalize: bool = True,
                                 apply_colormap: bool = True, 
                                 codec: str = "MJPG") -> Optional[str]:
        """
        Create a movie directly from a 3D frames array (t, h, w) and save under results/.
        The movie is saved as a AVI file.
        OpenCV is used to create the movie.
        
        Args:
            frames (np.ndarray): 3D frames array (t, h, w)
            output_path (str): Output path for the movie
            fps (int): Frames per second
            normalize (bool): Normalize the frames
            apply_colormap (bool): Apply a colormap to the frames
            codec (str): Codec for the movie
        Returns:
            Optional[str]: Output path if saved successfully, else None
        """
        
        
        t_len, height, width = frames.shape
        
        output_path = os.path.join(self.results_dir, f"dicom_movie.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        if not video.isOpened():
            logger.error("Failed to open video writer")
            return None
            
        # Determine normalization range
        if normalize:
            min_val = int(frames.min())
            max_val = int(frames.max())
            
        # Create the movie by traversing through the frames array in the time dimension.
        for t in range(t_len):
            arr = frames[t]
            
            if normalize:
                norm = ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)
 
            # Available colormaps: https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html?utm_source=chatgpt.com
            if apply_colormap:
                frame = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            else:
                frame = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
                
            video.write(frame)
            
        video.release()
        
        logger.info(f"Movie saved: {output_path}")
        
        return output_path


    def plot_pixel_over_time(self,
                             series: np.ndarray,
                             title: str = "Pixel Timeseries",
                             y_label: str = "Signal Intensity",
                             output_filename: str = "pixel_timeseries.png") -> Optional[str]:
        """
        Convenience method: extract a pixel's associated value over time and plot it.

        Args:
            series (np.ndarray): 1D array with shape (time, value)
            title (Optional[str]): Figure title"
            y_label (str): Y-axis label
            output_filename (str): Output filename in results directory

        Returns:
            Optional[str]: Output path if saved successfully, else None
        """
        
        if series is None:
            logger.error("plot_series: series is None")
            return None
        if not isinstance(series, np.ndarray):
            try:
                series = np.asarray(series)
            except Exception as e:
                logger.error(f"plot_series: cannot convert to np.ndarray: {e}")
                return None
        if series.ndim != 1:
            logger.error("plot_series: series must be 1D")
            return None

        os.makedirs(self.results_dir, exist_ok=True)
        output_path = os.path.join(self.results_dir, output_filename)

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(series, 'b-', linewidth=2)
            plt.title(title, fontsize=16)
            plt.xlabel('Time Frame', fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Series plot saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"plot_series: failed to save plot: {e}")
            return None


    def plot_mask(self, mask: np.ndarray, output_filename: str = "mask.png") -> Optional[str]:
        """
        Plot and save a 2D mask as a PNG file. If mask_index is provided, it is ignored for 2D masks (kept for compatibility).
        
        Args:
            mask (numpy.ndarray): Binary mask in (h, w) shape to plot.
            output_filename (str): Name of the file to save the plot.
        """
        if mask is None:
            logger.error("plot_mask: mask is None")
            return None
        if not isinstance(mask, np.ndarray):
            try:
                mask = np.asarray(mask)
            except Exception as e:
                logger.error(f"plot_mask: cannot convert to np.ndarray: {e}")
                return None

        if mask.ndim != 2:
            logger.error("plot_mask: mask must be a 2D array (h, w)")
            return None

        os.makedirs(self.results_dir, exist_ok=True)
        output_file = os.path.join(self.results_dir, output_filename)

        # Create the plot
        plt.figure(figsize=(10, 8))
        
        mask_to_plot = (1 - mask) * 255
        # Plot the mask
        im = plt.imshow(mask, cmap='gray_r', interpolation='nearest')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Mask plot saved to: {output_file}")
        plt.close()
        return output_file


if __name__ == "__main__":
    from data_loader import DataLoader
    data_loader = DataLoader()
    frames = data_loader.dicom()
    
    manager = SaveDataManager()
    manager.create_movie_from_frames(frames)
    
    # Example: plot intensity over time for the center pixel
    t_len, h, w = frames.shape
    center_row, center_col = h // 2, w // 2
    series = frames[:, center_row, center_col].astype(np.float64)
    
    manager.plot_pixel_over_time(series,
                                 title=f"Center Pixel ({center_row}, {center_col}) Intensity",
                                 y_label="Signal Intensity",
                                 output_filename="center_pixel_timeseries.png")
    
    
    mask = DataLoader(mask_file="input_data/AIF_And_Myo_Masks.tiff").mask(mask_index=0)
    manager.plot_mask(mask, output_filename="blood_pool.png")
    
    mask = DataLoader(mask_file="input_data/AIF_And_Myo_Masks.tiff").mask(mask_index=1)
    manager.plot_mask(mask, output_filename="myocardium.png")










