#main for for ANC Final Project.
#Creates, trains, and calls a CNN using the tensorflow library.

#This CNN is based off the research/work discussed in the following paper:
#"An Improved Performance of Deep Learning Based on Convolution Neural Network to Classify the Hand Motion by Evaluating Hyper Parameter"
#by Triwiyanto Triwiyanto, Member IEEE, I Putu Alit Pawana, and Mauridhi Hery Purnomo, Senior Member IEEE


#Library/Module Imports
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, GlobalAveragePooling1D, Dropout

#Custom Code Imports
from preprocess_segmentation import test_func


INPUT_CHANN_COUNT = 2
WINDOW_SIZE = 800
OVERLAP = 100

CNN_FILTER_COUNT = 200
KERNEL_SIZE = 8
STRIDE_LENGTH = 1
DROPOUT_RATE = 0.5
OUTPUT_CLASSES_COUNT = 10

def main():

    #----Load Data----#
    # Read datafile into dataframe
    f = 'compiled_data.csv' #compiled dataset filename
    path = '../data/' #update with path for locating compiled datset
    data = pd.read_csv(str(f), header = 0) #data = pd.read_csv(str(path) + '\\' + str(f), header = 0)
    data.head(5)

    #----Data Preprocessing----#
    # Split into test and train datasets
    train_data, test_data = train_test_datasets(data, 0.8)

    # Normalize train dataset
    train_data = normalize(train_data)

    # Create segments
    X_train, Y_train = segmentation(train_data, WINDOW_SIZE, OVERLAP)

    # Encode labels as binary vectors
    Y_train_encoded = label_encode(Y_train)

    #----Create CNN----#
    EMG_CNN = Sequential(name="EMG_CNN")

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
    print("Training EMG_CNN on data...FILENAME...[%Training Data]")
    history = EMG_CNN.fit(X_train, Y_train_encoded)

    #----Run CNN----#



if __name__ == "__main__":
    main()