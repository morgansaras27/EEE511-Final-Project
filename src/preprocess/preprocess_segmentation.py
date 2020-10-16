import pandas as pd
import numpy as np
from scipy import stats


'''SPLITS DATASET INTO TRAINING AND TESTING SETS'''
def train_test_datasets(df):
    # Differentiate between test set and training set = 96001
    df_test = df[df['user-id'] > 1000000]
    df_train = df[df['user-id'] <= 1000000]
    return df_train, df_test

'''NORMALIZE FEATURES'''
def normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    normalized_dataset = (dataset - mu)/sigma
    return normalized_dataset

'''CREATES SEGMENTS FROM DATASET'''
def segmentation(df, window_size, overlap):
    ''' 
    INPUTS 
        dataset = compiled dataset in a pandas dataframe
        window_size = size of segmentation window
        overlap = number of points of overlap between windows
    '''
    segments = [] #initialize empty list
    labels = [] #initialize empty list
    num_data_pts = np.shape(df)[0]
    step = window_size - overlap
    for i in range(0, num_data_pts-window_size, step):
        y_vals = df['y-axis'].values[i: i + window_size]
        z_vals = df['z-axis'].values[i: i + window_size]
        label = stats.mode(df['activity'][i: i + window_size])[0][0]
        segments.append([y_vals, z_vals])
        labels.append(label)
     # Outputs:list of individualsegments from compiled dataset
    return segments, labels

'''PUTTING IT ALL TOGETHER'''
# Read datafile into dataframe
f = 'Untitled Folder 1\compiled_data.csv' #compiled dataset filename
path = '' #update with path for locating compiled datset
data = pd.read_csv(str(f), header = 0) #data = pd.read_csv(str(path) + '\\' + str(f), header = 0)
data.head(5)

# Split into test and train datasets
train_data, test_data = train_test_datasets(data)
print('Train, test datasets split complete.')

# Normalize train dataset
train_data = normalize(train_data)
print('Normalized training dataset.')

# Create segments
window_size = 800
overlap = 100
time_steps = window_size
X_train, Y_train = segmentation(train_data, window_size, overlap)
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")
print('Shape of input dataset: ', np.shape(train_data))
print('Shape of segmented data:', np.shape(segments))
print('Shape of labels for segmented data: ', np.shape(labels))