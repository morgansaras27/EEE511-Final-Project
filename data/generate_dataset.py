'''
Name: generate_dataset.py
Description: Compile all data files into 1 single file
Author: Vu Phan
Date: 2020/10/10
'''

import os
import pandas as pd
import numpy as np

''' Macros '''
LABELS = ("CLO", # hand close    | HC-*
          "IND", # index         | I-I*
          "LIT", # little        | L-L*
          "MID", # middle        | M-M*
          "RIN", # ring          | R-R*
          "THI", # thumb-index   | T-I*
          "THL", # thumb-little  | T-L*
          "THM", # thumb-middle  | T-M*
          "THR", # thumb-ring    | T-R*
          "THU") # thumb         | T-T*
DATA_LENGTH = 20000 # 20,000 samples for each file
f = 2000
SUBJECT_PATH = ('EMG-S1', 
                'EMG-S2',
                'EMG-S3',
                'EMG-S4',
                'EMG-S5',
                'EMG-S6',
                'EMG-S7',
                'EMG-S8',
                'EMG-S9',
                'EMG-S10')

''' Utils functions '''
def get_activity(file_name):
    global LABELS, DATA_LENGTH
    
    # Check the activity
    if file_name[0:3] == 'HC-':
        label_idx = 0
    elif file_name[0:3] == 'I-I':
        label_idx = 1
    elif file_name[0:3] == 'L-L':
        label_idx = 2
    elif file_name[0:3] == 'M-M':
        label_idx = 3
    elif file_name[0:3] == 'R-R':
        label_idx = 4
    elif file_name[0:3] == 'T-I':
        label_idx = 5
    elif file_name[0:3] == 'T-L':
        label_idx = 6
    elif file_name[0:3] == 'T-M':
        label_idx = 7
    elif file_name[0:3] == 'T-R':
        label_idx = 8
    elif file_name[0:3] == 'T-T':
        label_idx = 9

    # Generate list of activity
    activity = [LABELS[label_idx]]*DATA_LENGTH
    
    return activity

''' Initialization '''
df = pd.DataFrame() # create empty dataframe

''' Complile data '''
# Time stamp is the same for all files so I just created one 
#timestamp = np.arange(1, DATA_LENGTH+1, 1); # from 1 to 20,000 # <- never mind, old version

# Loop through all subjects
for i in range(10):
    print('*** SUBJECT ' + str(i+1))
    #user_id = (i+1)*np.ones(DATA_LENGTH) # from 1 to 10 for 10 participants # <- never mind, old version
    path = SUBJECT_PATH[i]
    files = os.listdir(path)

    # Pick only .csv file (redundant code)
    csv_files = [f for f in files if f[-3:] == 'csv']

    # Loop through all data files of each subject
    for f in csv_files:
        # Read EMG data
        data = pd.read_csv(str(path) + '\\' + str(f), header = None)
        # Add user-id, activity, and timestamp to the data
        #data.insert(0, 'timestamp', timestamp, True) # <- never mind, old version
        activity = get_activity(f) # get activity label according to the file name
        data.insert(0, 'activity', activity, True)
        #data.insert(0, 'user-id', user_id, True) # <- never mind, old version

        # Add to the compiled file
        df = df.append(data)
        print('-> Finished ' + str(f))
        #break # for debugging purpose, get 1 file only

''' Export '''
# Add user-id and timestamp
user_id = np.arange(1, df.shape[0]+1, 1)
timestamp = user_id/2000.0
df.insert(1, 'timestamp', timestamp, True)
df.insert(0, 'user-id', user_id, True)
# Re-format the data frame labels
df.columns = ['user-id', 'activity', 'timestamp', 'y-axis', 'z-axis']
# Export to a single .csv file
print('Started exporting the .csv file')
df.to_csv('compiled_data.csv', index = False)
print('Finished exporting the .csv file')
print('--------------------------------')
