import pandas as pd
import numpy as np
from scipy import stats

'''SPLITS DATASET INTO TRAINING AND TESTING SETS'''
def train_test_datasets(X, Y, percent_train):
    from sklearn.model_selection import train_test_split
    
    # Differentiate between test set and training set = 96001
    length = np.shape(X)[0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-percent_train)
    print('X_train shape (should be (9593, 800, 2)): ', np.shape(X_train))
    print('Y_train shape (should be (9593, 10)): ', np.shape(Y_train))
    
    return X_train, Y_train, X_test, Y_test

'''NORMALIZE FEATURES'''
def normalize(dataset):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    mu_y = np.mean(dataset['y-axis'], axis=0)
    mu_z = np.mean(dataset['z-axis'], axis=0)
    
    sigma_y = np.std(dataset['y-axis'], axis=0)
    sigma_z = np.std(dataset['z-axis'], axis=0)

    normalized_y = (dataset['y-axis'].to_numpy() - mu_y) / sigma_y
    normalized_z = (dataset['z-axis'].to_numpy() - mu_z) / sigma_z
    
    dataset.loc[:,('y-axis')] = normalized_y
    
    dataset.loc[:,('z-axis')] = normalized_z
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
    step = overlap
    for i in range(0, num_data_pts-window_size, step):
        y_vals = df['y-axis'].values[i: i + window_size]
        z_vals = df['z-axis'].values[i: i + window_size]
        if (np.size(np.unique(df['activity'][i: i + window_size])[0]) == 1): #1 unique label in segement
            label = stats.mode(df['activity'][i: i + window_size])[0][0]
            segments.append([y_vals, z_vals])
            labels.append(label)
    
    # Reshaping
    segments = np.asarray(segments, dtype= np.float32).reshape(-1, window_size, 2)
    labels = np.asarray(labels)
    print('Shape of segmented dataset (should be (11992, 800, 2)):', np.shape(segments))
    print('Shape of labels for segmented data (should be (11992,)): ', np.shape(labels))
    
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
    
    return onehot_encoded

'''SHUFFLE DATA'''
def shuffle_data(X, Y):
    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y)
    return X, Y

''' OVERALL FUNCTION TO PRE-PROCESS DATA '''
# Calls all other pre-processing functions in correct order, outputs X,Y for train and test
def preprocess(df, percent_train, window_size, overlap, subject):
    ''' INPUTS:
        df = compiled dataframe
        percent_train = percentage of dataset that is used for training
        window_size = window for segmentation
        overlap = overlap for segments
        subject = subject number 0 - 9 (for subjects 1 - 10) '''
    
    # Create data subset for given subject
    data = df[df['user-id'] > (1200000*subject)]
    data = data[data['user-id'] <= (1200000*(subject+1))]
    print('Shape of subject %i data (should be (1200000,5)):' % (subject))
    print(np.shape(data))
    
    # Normalize entire dataset
    data = normalize(data)

    # Create segments for train and test datasets
    X, Y = segmentation(data, window_size, overlap)
    
    # Convert to float for Keras to process data                   
    X = X.astype("float32")

    # Encode labels as binary vectors for train and test datasets
    Y_encoded = label_encode(Y)
    
    # Shuffle data
    X, Y_encoded = shuffle_data(X, Y_encoded)
    
    # Split into test and train datasets
    X_train, Y_train_encoded, X_test, Y_test_encoded = train_test_datasets(X, Y_encoded, percent_train)
    
    ''' OUTPUTS
        X's are normalized features for given subject
        Y's are binary vector encoded labels for gestures '''

    
    return X_train, Y_train_encoded, X_test, Y_test_encoded



n=1
if n == 1:
    '''PUTTING IT ALL TOGETHER'''
    f = 'compiled_data.csv' #compiled dataset filename
    path = '' #update with path for locating compiled datset
    df = pd.read_csv(str(f), header = 0) #data = pd.read_csv(str(path) + '\\' + str(f), header = 0)
    df.head(5)

    percent_train = 0.8
    window_size = 800
    overlap = 100

    # Initialize dictionaries for X and Y for train and test, key will be subject # 0-9, value will be the segments or labels
    X_train = dict()
    Y_train = dict()
    X_test = dict()
    Y_test = dict()

    for i in range(0, 10, 1):
        subject = i
        x_train, y_train, x_test, y_test = preprocess(df, percent_train, window_size, overlap, subject)
        X_train[i] = x_train
        Y_train[i] = y_train
        X_test[i] = x_test
        Y_test[i] = y_test
        print('--------------------- SUBJECT %i COMPLETE ----------------\n\n' % subject)
