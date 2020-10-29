#main for for ANC Final Project.
#Creates, trains, and calls a CNN using the tensorflow library.

#This CNN is based off the research/work discussed in the following paper:
#"An Improved Performance of Deep Learning Based on Convolution Neural Network to Classify the Hand Motion by Evaluating Hyper Parameter"
#by Triwiyanto Triwiyanto, Member IEEE, I Putu Alit Pawana, and Mauridhi Hery Purnomo, Senior Member IEEE


#Library/Module Imports
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Reshape, Dense, Conv1D, MaxPool1D, GlobalAveragePooling1D, Dropout

#Custom Code Imports
from preprocess_segmentation import *

NUM_SUBJECTS = 10
INPUT_CHANN_COUNT = 2
WINDOW_SIZE = 800
OVERLAP = 100

CNN_FILTER_COUNT = 200
KERNEL_SIZE = 8
STRIDE_LENGTH = 1
DROPOUT_RATE = 0.5
OUTPUT_CLASSES_COUNT = 10

PERCENT_TRAINING = 0.8

def main():

    #----Load Data----#
    # Read datafile into dataframe
    f = 'compiled_data.csv' #compiled dataset filename
    path = '../data/' #update with path for locating compiled datset
    data = pd.read_csv(str(path+f), header = 0) #data = pd.read_csv(str(path) + '\\' + str(f), header = 0)
    data.head(5)

    #----Data Preprocessing----#
    # Split into test and train datasets
    # Initialize dictionaries for X and Y for train and test
    # Key will be subject # 0-NUM_SUBJECTS, value will be the segments or labels
    X_train = dict()
    Y_train = dict()
    X_test = dict()
    Y_test = dict()

    for i in range(NUM_SUBJECTS):
        subject_index = i
        x_train, y_train, x_test, y_test = preprocess(data, PERCENT_TRAINING, WINDOW_SIZE, OVERLAP, subject_index)
        X_train[i] = x_train
        Y_train[i] = y_train
        X_test[i] = x_test
        Y_test[i] = y_test

    #----Create CNNs----#
    for cnn_index in range(1): #Only 1 CNN is created for a single subject, currently
        EMG_CNN = Sequential(name="EMG_CNN"+str(cnn_index))
        
        EMG_CNN.add(Reshape((WINDOW_SIZE, INPUT_CHANN_COUNT), input_shape=(1600,)))
        EMG_CNN.add(Conv1D(filters=CNN_FILTER_COUNT, kernel_size=KERNEL_SIZE, activation='relu', strides=STRIDE_LENGTH, input_shape=(WINDOW_SIZE,INPUT_CHANN_COUNT)))  #Conv
        EMG_CNN.add(Conv1D(filters=CNN_FILTER_COUNT, kernel_size=KERNEL_SIZE, activation='relu', strides=STRIDE_LENGTH))                       #Conv
        EMG_CNN.add(MaxPool1D(pool_size=8, strides=8))                                                                                         #Max Pooling
        EMG_CNN.add(Conv1D(filters=CNN_FILTER_COUNT, kernel_size=KERNEL_SIZE, activation='relu', strides=STRIDE_LENGTH))                       #Conv
        EMG_CNN.add(Conv1D(filters=CNN_FILTER_COUNT, kernel_size=KERNEL_SIZE, activation='relu', strides=STRIDE_LENGTH))                       #Conv
        EMG_CNN.add(GlobalAveragePooling1D())                                                                                                  #Global Avg. Pooling
        EMG_CNN.add(Dropout(DROPOUT_RATE))                                                                                                     #Dropout
        EMG_CNN.add(Dense(OUTPUT_CLASSES_COUNT, activation='relu'))                                                                            #Fully Connected

        EMG_CNN.compile(loss='categorical_crossentropy',
                        optimizer='Adagrad', 
                        metrics=['accuracy'])

        print(EMG_CNN.summary())

        #----Train CNN----#
        # print("Training EMG_CNN on data...FILENAME...[%Training Data]")
        # history = EMG_CNN.fit(X_train[cnn_index], Y_train[cnn_index])
        print(EMG_CNN.predict(np.stack(X_train[0][0]).flatten))
    #----Run CNN----#



if __name__ == "__main__":
    main()