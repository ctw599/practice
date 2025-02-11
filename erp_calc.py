import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import math

def calc_mean_erp(data, pts):
    pts = pts.astype(int)
    # time before event in ms
    preonset = 200  
    # time after event in ms
    postonset = 1000  
    epoch_size = preonset + postonset + 1
    tvec = np.arange(-preonset, postonset + 1) 
    
    # Initialize ERP array
    erp = np.zeros((epoch_size, pts.shape[0]))
    # Calculate ERPs for each event
    # print('Calculating ERPs...')
    for k in range(pts.shape[0]):
        erp[:,k] = data[pts[k, 0] - preonset:pts[k, 0] + postonset +1].reshape(-1)
    
    # Average ERP for each movement type
    nmovement = np.unique(pts[:, 2])  # unique movement types
    erp_mean = np.zeros((epoch_size, 5))
    
    for n, movement in enumerate(nmovement):
        inds = pts[:, 2] == movement  # indices for trials of the current movement type
        erp_mean[:,n] = np.mean(erp[:, inds], axis=1)
    
    # Plot un-smoothed ERP for movement types 1 and 3 (example)
    # plt.figure()
    # plt.plot(tvec, erp_mean[:, [0, 4]])  # example: channel 6, movement types 1 and 3
    # # plt.axis([-200, 1000, -15, 15])
    # plt.title("ERP for Movement Types 1 and 3 (Channel 6)")
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Amplitude (uV)')
    # plt.show()
    # plt.close()

    return erp_mean


mat_file = r'C:\Users\user\Downloads\fingerflex\jp\jp_fingerflex.mat'  # Replace with the actual path to your .mat file

data = scipy.io.loadmat(mat_file)
ecog_data = pd.read_csv(r'C:\Users\user\OneDrive - Bar-Ilan University - Students\Documents\advanced_programming_course\time_series\mini_project_2_data\brain_data_channel_one.csv', header=None)  # No header, assuming raw EEG data
ecog_data = ecog_data.to_numpy()

data_df = pd.read_csv(r'C:\Users\user\OneDrive - Bar-Ilan University - Students\Documents\advanced_programming_course\time_series\mini_project_2_data\events_file_ordered.csv', header=None)  # No header, assuming raw EEG data
event_points = data_df.to_numpy()  # Convert to NumPy array
# Call the function
erp_mean = calc_mean_erp(ecog_data, event_points)
# mean_of_mean = np.mean(erp_mean, axis=1)
# def row_average(row):
#     return row.mean()
# erp_mean = np.transpose(erp_mean)
# print(erp_mean)
# mean_of_mean = np.mean(erp_mean, axis=1)
# print(mean_of_mean.mean())
# print(mean_of_mean)
# for mini_list in erp_mean:
#     print(row_average(mini_list))