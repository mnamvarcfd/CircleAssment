from data_loader import DataLoader
from save_data_manager import SaveDataManager
from compute_quantity import ComputeQuantity
from myocardial_blood_flow import MyocardialBloodFlow
import pandas as pd
import numpy as np

#=============================================Load Data===========================================
data_loader = DataLoader(dicom_dir="input_data/DICOM_files")
frames = data_loader.dicom()

blood_pool_mask = data_loader.mask(mask_index=0)

myocardium_mask = data_loader.mask(mask_index=1)

#=============================================Save Data===========================================
save_data_manager = SaveDataManager()

# Create movie from frames
save_data_manager.create_movie_from_frames(frames)

# plot intensity over time for the center pixel
t_len, h, w = frames.shape
center_row = h // 2
center_col = w // 2
series = frames[:, center_row, center_col].astype(np.float64)
save_data_manager.plot_pixel_over_time(series,
                                title=f"Center Pixel ({center_row}, {center_col}) Intensity",
                                y_label="Signal Intensity",
                                output_filename="center_pixel_timeseries.png")


# plot blood pool region
save_data_manager.plot_mask(blood_pool_mask, output_filename="blood_pool.png")

# plot myocardium region
save_data_manager.plot_mask(myocardium_mask, output_filename="myocardium.png")

# =============================================Compute Quantity===========================================
compute_quantity = ComputeQuantity(frames=frames, aif_mask=blood_pool_mask, myo_mask=myocardium_mask)

#compute AIF
aif =compute_quantity.arterial_input_function()

#plot AIF
save_data_manager.plot_pixel_over_time(aif,
                                       title=f"Arterial Input Function (AIF)",
                                       y_label="Signal Intensity",
                                       output_filename="arterial_input_function.png")

#compute myocardium time series
myo_pixel_coordinates, myo_time_series = compute_quantity.myocardium_time_series()

#plot myocardium time series at 10th pixel
save_data_manager.plot_pixel_over_time(myo_time_series[10],
                                       title=f"MYOcardium time series (MYO) at 10th pixel",
                                       y_label="Signal Intensity",
                                       output_filename="myocardium_time_series_10.png")

# =============================================Compute MBF===========================================
myocardial_blood_flow = MyocardialBloodFlow(aif=aif, myo=myo_time_series.T)

mbf = myocardial_blood_flow.compute()

save_data_manager.save_image(mbf, myocardium_mask, Value_title="MBF", output_filename="mbf_map.png")

# Save MBF results to CSV
mbf_df = pd.DataFrame(mbf, columns=['MBF'])
mbf_df.to_csv('results/mbf_results.csv', index=False)
