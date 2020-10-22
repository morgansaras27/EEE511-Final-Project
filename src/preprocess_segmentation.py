import pandas as pd
import numpy as np
from scipy import stats

'''SPLITS DATASET INTO TRAINING AND TESTING SETS'''
def train_test_datasets(df, percent_train):
    # Differentiate between test set and training set = 96001
    length = np.shape(df)[0]
    df_test = df[df['user-id'] > length*percent_train]
    df_train = df[df['user-id'] <= length*percent_train]
    print('Train-test dataset split complete.')
    
    return df_train, df_test

'''NORMALIZE FEATURES'''
def normalize(dataset):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    mu_y = np.mean(dataset['y-axis'], axis=0)
    mu_z = np.mean(dataset['z-axis'], axis=0)
    print('\nAverages for y-axis, z-axis = ', mu_y, mu_z)
    
    sigma_y = np.std(dataset['y-axis'], axis=0)
    sigma_z = np.std(dataset['z-axis'], axis=0)
    print('Std. Deviations = ', sigma_y, sigma_z)
    
    normalized_y = (dataset['y-axis'].to_numpy() - mu_y) / sigma_y
    normalized_z = (dataset['z-axis'].to_numpy() - mu_z) / sigma_z
    
    dataset.loc[:,('y-axis')] = normalized_y
    print('Replaced with normalized y-values.')
    
    dataset.loc[:,('z-axis')] = normalized_z
    print('Replaced with normalized z-values.')
    dataset.head(5)
    
    return dataset

'''CREATES SEGMENTS FROM DATASET'''
def segmentation(df, window_size, overlap):
    ''' INPUTS: 
        df = compiled dataset in a pandas dataframe
        window_size = size of segmentation window
        overlap = number of points of overlap between window '''
    
    segments = [] #initialize empty list
    labels = [] #initialize empty list
    num_data_pts = np.shape(df)[0]
    step = window_size - overlap
    for i in range(0, num_data_pts-window_size, step):
        y_vals = df['y-axis'].values[i: i + window_size]
        z_vals = df['z-axis'].values[i: i + window_size]
        if (np.size(np.unique(df['activity'][i: i + window_size])[0]) == 1): #1 unique label in segement
            label = stats.mode(df['activity'][i: i + window_size])[0][0]
            segments.append([y_vals, z_vals])
            labels.append(label)
    
    print('\nSegmentation complete. ')
    print('Shape of input dataset: ', np.shape(df))
    print('Shape of segmented data:', np.shape(segments))
    print('Shape of labels for segmented data: ', np.shape(labels))
    
    
    ''' OUTPUTS:
        segments = list of individual segments from compiled dataset
        labels = list of labels for each segment from compiled dataset '''
    return segments, labels

''' ENCODE LABELS INTO BINARY VECTORS '''
def label_encode(labels):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    
    # encode as integers
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    
    # encode as vectors of 1's and 0's (one-hot)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    print('\nOnehot encoding complete. Shape of output vectors:', np.shape(onehot_encoded))
    
    return onehot_encoded


if __name__ == "__main__":
    '''PUTTING IT ALL TOGETHER'''
    # Read datafile into dataframe
    f = 'Untitled Folder 1\compiled_data.csv' #compiled dataset filename
    path = '' #update with path for locating compiled datset
    data = pd.read_csv(str(f), header = 0) #data = pd.read_csv(str(path) + '\\' + str(f), header = 0)
    data.head(5)

    # Split into test and train datasets
    train_data, test_data = train_test_datasets(data, 0.8)

    # Normalize train dataset
    train_data = normalize(train_data)

    # Create segments
    window_size = 800
    overlap = 100
    time_steps = window_size
    X_train, Y_train = segmentation(train_data, window_size, overlap)

    # Encode labels as binary vectors
    Y_train_encoded = label_encode(Y_train)