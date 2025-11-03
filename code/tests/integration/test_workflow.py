"""
Integration tests for the complete myocardial blood flow computation workflow.
"""
import numpy as np
import pytest
import os
import pandas as pd
from data_loader import DataLoader
from compute_quantity import ComputeQuantity
from myocardial_blood_flow import MyocardialBloodFlow
from save_data_manager import SaveDataManager


class TestWorkflow:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self, temp_dir):
        """Test the complete workflow from data loading to MBF computation and saving."""

        #=============================================Load Data===========================================
        data_loader = DataLoader(dicom_dir="input_data/DICOM_files")
        frames = data_loader.dicom()

        blood_pool_mask = data_loader.mask(mask_index=0)

        myocardium_mask = data_loader.mask(mask_index=1)


        #=============================================Save Data===========================================
        save_data_manager = SaveDataManager()

        # Create movie from frames
        save_data_manager.create_movie_from_frames(frames)

        # plot blood pool region
        save_data_manager.plot_mask(blood_pool_mask, output_filename="blood_pool.png")

        # plot myocardium region
        save_data_manager.plot_mask(myocardium_mask, output_filename="myocardium.png")


        # =============================================Compute Quantity===========================================
        compute_quantity = ComputeQuantity(frames=frames, blood_pool_mask=blood_pool_mask, myo_mask=myocardium_mask)

        #compute blood pool pixel time series
        blood_pool_pixel_coordinates, blood_pool_time_series = compute_quantity.blood_pool_time_series()

        #plot blood pool time series at 10th pixel
        save_data_manager.plot_pixel_over_time(blood_pool_time_series[10],
                                            title=f"Blood Pool time series at 10th pixel",
                                            y_label="Signal Intensity",
                                            output_filename="blood_pool_time_series_10.png")

        #compute myocardium time series
        myo_pixel_coordinates, myo_time_series = compute_quantity.myocardium_time_series()

        #plot myocardium time series at 10th pixel
        save_data_manager.plot_pixel_over_time(myo_time_series[10],
                                            title=f"MYOcardium time series (MYO) at 10th pixel",
                                            y_label="Signal Intensity",
                                            output_filename="myocardium_time_series_10.png")

        #compute AIF
        aif =compute_quantity.arterial_input_function()

        #plot AIF
        save_data_manager.plot_pixel_over_time(aif,
                                            title=f"Arterial Input Function (AIF)",
                                            y_label="Signal Intensity",
                                            output_filename="arterial_input_function.png")

        # =============================================Compute MBF===========================================
        # Reconstruct 3D myocardium array from time series for MyocardialBloodFlow
        myo_3d = np.zeros((frames.shape[0], myocardium_mask.shape[0], myocardium_mask.shape[1]))
        for i, (y, x) in enumerate(myo_pixel_coordinates):
            myo_3d[:, y, x] = myo_time_series[i, :]

        myocardial_blood_flow = MyocardialBloodFlow(aif=aif, myo=myo_3d, myo_mask=myocardium_mask)

        mbf = myocardial_blood_flow.compute()

        save_data_manager.save_image(mbf, myocardium_mask, Value_title="MBF", output_filename="mbf_map_taud_001.png")

        # Save MBF results to CSV (flatten the 2D array to save pixel values)
        mbf_flat = mbf.flatten()
        mbf_df = pd.DataFrame({'MBF': mbf_flat})
        mbf_df.to_csv('results/mbf_results.csv', index=False)
