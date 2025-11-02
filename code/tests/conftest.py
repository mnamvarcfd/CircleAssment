"""
Shared pytest fixtures for testing.
"""
import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import tempfile
import shutil    
from scipy.signal import convolve
    

# Add parent directory to path to allow importing src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


num_frames = 60
image_size = 50

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


def fermi(t:np.ndarray, F:float, tau_0:float, k:float)->np.ndarray:
    """
    Fermi function for impulse response
    The description of the args is based on eq 5 in Jerosch-Herold 1998 paper
    tau_d (float): It interperited as the delay of the impulse response function.
    
    Args:
        t (numpy.ndarray): Time
        F (float): Rate of flow
        tau_0 (float): width of the shoulder of the Fermi function
        k (float): decay rate of Fermi function due to contrast agent washout.
    Returns:
        R_F (numpy.ndarray): Impulse response function (Fermi function)
    """
    tau_d = 1
    
    delayed_t = t - tau_d

    step_function = (delayed_t >= 0).astype(np.float64)
  
    exponent = np.exp(k * (delayed_t - tau_0)) + 1.0
   
    R_F = F / exponent * step_function
    
    return R_F



@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    
    Files are written directly to tests/test_outputs/ directory.
    Files will remain after the test for inspection.
    Files are kept after the test for inspection.
    You can manually delete them if needed.
    """

    test_outputs_dir = os.path.join(project_root, 'tests', 'test_outputs')
    os.makedirs(test_outputs_dir, exist_ok=True)
    
    yield test_outputs_dir


@pytest.fixture
def sample_blood_pool_mask():
    """Create a sample 2D binary mask for blood pool region.
    
    Returns a binary mask with a square region set to 1.
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Create a square region in the center
    center_y, center_x = image_size // 2, image_size // 2
    blood_pool_size = image_size // 4  
    
    y, x = np.ogrid[:image_size, :image_size]
    
    # Create square region: both x and y must be within half_size from center
    mask[(np.abs(x - center_x) <= blood_pool_size/2) & (np.abs(y - center_y) <= blood_pool_size/2)] = 1

    return mask


@pytest.fixture
def sample_myocardium_mask():
    """Create a sample 2D binary mask for myocardium region.
    
    Returns a binary mask padded around the blood pool region.
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Create a square region in the center
    center_y, center_x = image_size // 2, image_size // 2
    blood_pool_size = image_size // 4
    myocardium_size = image_size // 4  
    
    y, x = np.ogrid[:image_size, :image_size]
    
    # Create larger square for myocardium
    mask[(np.abs(x - center_x) <= myocardium_size/2+blood_pool_size/2) & (np.abs(y - center_y) <= myocardium_size/2+blood_pool_size/2)] = 1

    # Remove blood pool region from myocardium mask
    mask[(np.abs(x - center_x) <= blood_pool_size/2) & (np.abs(y - center_y) <= blood_pool_size/2)] = 0

    return mask


@pytest.fixture
def sample_blood_pool_data(sample_blood_pool_mask):
    """Create sample data for blood pool time series."""
    blood_pool_mask = sample_blood_pool_mask
    
    blood_pool_data = np.zeros((num_frames, image_size, image_size), dtype=np.float64)
    
    y_coords, x_coords = np.where(blood_pool_mask == 1)
    
    time_series = gamma_variate(np.arange(num_frames), init_value=10, A=150, alpha=3.5, beta=4.5)
    
    for y, x in zip(y_coords, x_coords):
        blood_pool_data[:, y, x] = time_series
        
    return blood_pool_data


@pytest.fixture
def sample_aif(sample_blood_pool_data):
    """Create a sample Arterial Input Function (AIF) time series."""

    aif = np.mean(sample_blood_pool_data, axis=(1, 2))

    return aif.astype(np.float64)


@pytest.fixture
def sample_tissue_impulse_response(sample_myocardium_mask):
    """Create sample data for tissue impulse response time series."""

    tissue_impulse_response = np.zeros((num_frames, image_size, image_size), dtype=np.float64)
    
    y_coords, x_coords = np.where(sample_myocardium_mask == 1)
    
    time_series = fermi(t=np.arange(num_frames), F=1, tau_0=20, k=0.1)

    for y, x in zip(y_coords, x_coords):
        tissue_impulse_response[:, y, x] = time_series
        
    return tissue_impulse_response


@pytest.fixture
def sample_myocardium_data(sample_myocardium_mask, sample_aif, sample_tissue_impulse_response):
    """Create a sample myocardium data by convolving the AIF with tissue impulse response."""

    # Get pixel coordinates where mask == 1
    y_coords, x_coords = np.where(sample_myocardium_mask == 1)
        
    myocardium_data = np.zeros((num_frames, image_size, image_size), dtype=np.float64)
    
    # For each pixel, convolve its time series with AIF
    for i, (y, x) in enumerate(zip(y_coords, x_coords)):
        # Extract the time series for this pixel: shape (num_frames,)
        pixel_impulse_response = sample_tissue_impulse_response[:, y, x]
        
        # Convolve AIF with impulse response (as done in _convolution_model)
        # Use 'full' mode and take first len(sample_aif) elements to match time dimension
        convolved = convolve(sample_aif, pixel_impulse_response, mode='full')[:len(sample_aif)]
        myocardium_data[:, y, x] = convolved
    
    return myocardium_data


@pytest.fixture
def sample_frames(sample_blood_pool_data, sample_myocardium_data):
    """Create a sample 3D frames array for testing."""
    frames = sample_blood_pool_data + sample_myocardium_data
        
    return frames


