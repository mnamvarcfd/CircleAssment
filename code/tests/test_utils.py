"""
Shared utility functions for tests.
"""

import numpy as np
from scipy.signal import convolve
from myocardial_blood_flow_fixed_tau_d import MyocardialBloodFlow


def gamma_variate(t, init_value=0.0, A=1.0, alpha=2.0, beta=1.0):
    """
    Gamma-variate function with baseline shift (rise and decay curve).

    Args:
        t: time points
        init_value: value to shift the whole curve upward
        A: amplitude of the peak (scales the curve)
        alpha: controls rise steepness
        beta: controls decay speed
    """
    f = A * (t ** alpha) * np.exp(-t / beta)
    # Normalize peak to A
    f = A * f / np.max(f)
    # Shift the curve upward
    f = f + init_value
    return f


class GenerateDataForAnaliticalTest:
    def __init__(self):
        self.num_frames = 60
        self.image_size = 50

        self.blood_pool_mask = self.generate_blood_pool_mask()
        self.myocardium_mask = self.generate_myocardium_mask()
        
        self.tissue_impulse_response_time_series_fermi = self.tissue_impulse_response_time_series_fermi()
        
        self.blood_pool_time_series = self.blood_pool_time_series()
        self.myocardium_time_series = self.myocardium_time_series()
        
        self.frames = self.generate_frames()
    
    def generate_blood_pool_mask(self):
        """Create a sample 2D binary mask for blood pool region.

        Returns a binary mask with a square region set to 1.
        """
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Create a square region in the center
        center_y, center_x = self.image_size // 2, self.image_size // 2
        # Blood pool should be smaller - use 1/8 of image size for half-width
        blood_pool_half_width = self.image_size // 8

        y, x = np.ogrid[:self.image_size, :self.image_size]

        # Create square region: both x and y must be within half_width from center
        mask[(np.abs(x - center_x) <= blood_pool_half_width) & (np.abs(y - center_y) <= blood_pool_half_width)] = 1

        return mask

  
    def generate_myocardium_mask(self):
        """Create a sample 2D binary mask for myocardium region.

        Returns a binary mask padded around the blood pool region.
        """
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Create a square region in the center
        center_y, center_x = self.image_size // 2, self.image_size // 2
        blood_pool_half_width = self.image_size // 8
        myocardium_half_width = self.image_size // 4  # Myocardium extends further out

        y, x = np.ogrid[:self.image_size, :self.image_size]

        # Create larger square for myocardium (extends from blood pool edge to myocardium edge)
        mask[(np.abs(x - center_x) <= myocardium_half_width) & (np.abs(y - center_y) <= myocardium_half_width)] = 1

        # Remove blood pool region from myocardium mask
        mask[(np.abs(x - center_x) <= blood_pool_half_width) & (np.abs(y - center_y) <= blood_pool_half_width)] = 0

        return mask


    def blood_pool_time_series(self):
        """Create a sample time series for blood pool region."""
        time_series = gamma_variate(np.arange(self.num_frames), init_value=10, A=150, alpha=3.5, beta=4.5)
            
        return time_series


    def tissue_impulse_response_time_series_fermi(self):
        """Create sample data for tissue impulse response time series."""
        time_series = MyocardialBloodFlow.fermi_function(t=np.arange(self.num_frames), F=1, tau_0=20, k=0.1)

        return time_series


    def myocardium_time_series(self):
        """Create a sample time series for myocardium region."""

        # Convolve AIF with impulse response (as done in _convolution_model)
        # Use 'full' mode and take first len(blood_pool_time_series) elements to match time dimension
        convolved = convolve(self.blood_pool_time_series, self.tissue_impulse_response_time_series_fermi, mode='full')[:len(self.blood_pool_time_series)]

        return convolved

    def generate_frames(self):
        """Create a sample 3D frames array for testing."""
        frames = np.zeros((self.num_frames, self.image_size, self.image_size), dtype=np.float64)

        for t in range(self.num_frames):
            for x in range(self.image_size):
                for y in range(self.image_size):
                    if self.blood_pool_mask[y, x] == 1:
                        frames[t, y, x] = self.blood_pool_time_series[t]
                    elif self.myocardium_mask[y, x] == 1:
                        frames[t, y, x] = self.myocardium_time_series[t]
                    else:
                        frames[t, y, x] = 0

        return frames


    def aif_time_series(self):
        return self.blood_pool_time_series

