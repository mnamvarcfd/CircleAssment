"""
Integration tests for the complete myocardial blood flow computation workflow.
"""
import numpy as np
import pytest
import os
import pandas as pd
from data_loader import DataLoader
from compute_quantity import ComputeQuantity
from myocardial_blood_flow_fixed_tau_d import MyocardialBloodFlow
from save_data_manager import SaveDataManager


def dimensionalize_mbf(dimensionless_mbf_map: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Converts dimensionless MBF values to physical units of mL/g/min.
    
    According to the central volume principle and the assignment instructions:
    MBF_physical = (F_dimensionless / time_per_frame) * (60 sec/min) / (1.05 g/mL)
    
    Args:
        dimensionless_mbf_map: Array of dimensionless F values from curve fitting (shape: num_pixels,)
        time: Array of elapsed times in seconds (shape: num_frames,) - first frame should be 0.0
        
    Returns:
        np.ndarray: MBF values in mL/g/min (shape: num_pixels,)
    """
    # Calculate time intervals between consecutive frames
    if len(time) < 2:
        raise ValueError("Need at least 2 time points to calculate time intervals")
    
    # Calculate time differences between consecutive frames
    time_intervals = np.diff(time)
    
    # Filter out any invalid intervals (NaN or zero)
    valid_intervals = time_intervals[~np.isnan(time_intervals) & (time_intervals > 0)]
    
    if len(valid_intervals) == 0:
        raise ValueError("No valid time intervals found. Cannot calculate dimensional MBF.")
    
    # Use average time interval per frame (in seconds)
    avg_time_interval = np.mean(valid_intervals)
    
    # Calculate the scaling factor
    # Unit analysis: [1/sec] * [60 sec/min] / [g/mL] = mL/g/min
    # time_per_frame is the average interval between frames
    scaling_factor = 60.0 / (avg_time_interval * 1.05)
    
    # Apply scaling to all pixels (scaling_factor is a scalar, so it broadcasts correctly)
    dimensional_mbf = dimensionless_mbf_map * scaling_factor
    
    return dimensional_mbf


@pytest.mark.parametrize("tau_d", [0.1, 1.0, 5.0, 10.0])
def test_workflow_real(tau_d):
    """Test the complete workflow from data loading to MBF computation and saving."""

    #=============================================Load Data===========================================
    data_loader = DataLoader(dicom_dir="input_data/DICOM_files")
    frames = data_loader.dicom()

    blood_pool_mask = data_loader.mask(mask_index=0)

    myocardium_mask = data_loader.mask(mask_index=1)


    #=============================================Save Data===========================================
    save_data_manager = SaveDataManager(results_dir="tests/results/workflow_real_fixed_tau_d")

    # =============================================Compute Quantity===========================================
    compute_quantity = ComputeQuantity(frames=frames, blood_pool_mask=blood_pool_mask, myo_mask=myocardium_mask)

    #compute blood pool pixel time series
    blood_pool_pixel_coordinates, blood_pool_time_series = compute_quantity.blood_pool_time_series()

    #compute myocardium time series
    myo_pixel_coordinates, myo_time_series = compute_quantity.myocardium_time_series()

    #compute AIF
    aif =compute_quantity.arterial_input_function()

    # =============================================Compute MBF===========================================
    myocardial_blood_flow = MyocardialBloodFlow(frames=frames, blood_pool_mask=blood_pool_mask, myo_mask=myocardium_mask, tau_d=tau_d)

    mbf = myocardial_blood_flow.compute()

    save_data_manager.save_image(mbf, myocardium_mask, Value_title=f"MBF (tau_d={tau_d})", output_filename=f"non_scaled_mbf_map_tau_d_{tau_d}.png")

    time = data_loader.time
    dimensional_mbf = dimensionalize_mbf(mbf, time)
    save_data_manager.save_image(dimensional_mbf, myocardium_mask, Value_title=f"Scaled MBF (tau_d={tau_d})", output_filename=f"Scaled_MBF_map_tau_d_{tau_d}.png")



    # Save MBF results to CSV (flatten the 2D array to save pixel values)
    mbf_flat = mbf.flatten()
    mbf_df = pd.DataFrame({'MBF': mbf_flat, 'tau_d': tau_d})
    mbf_csv_path = os.path.join(save_data_manager.results_dir, f'mbf_results_tau_d_{tau_d}.csv')
    mbf_df.to_csv(mbf_csv_path, index=False)
