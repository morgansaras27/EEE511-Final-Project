#main for for ANC Final Project.
#Creates, trains, and calls a CNN using the tensorflow library.

#This CNN is based off the research/work discussed in the following paper:
#"An Improved Performance of Deep Learning Based on Convolution Neural Network to Classify the Hand Motion by Evaluating Hyper Parameter"
#by Triwiyanto Triwiyanto, Member IEEE, I Putu Alit Pawana, and Mauridhi Hery Purnomo, Senior Member IEEE


#Library/Module Imports
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Reshape, Dense, Conv1D, MaxPool1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adagrad
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
import seaborn as sns

#Custom Code Imports
from preprocess_segmentation import *


#Variables for input data
NUM_SUBJECTS = 10 # number of subjects being analyzed -- CHANGE BASED ON DATASET BEING USED
INPUT_CHANN_COUNT = 2 #number of input channels (always 2)
WINDOW_SIZE = 800 #window size used in segmentation, defines input size for CNN (always 800)
OVERLAP = 100 #overlap between windows used in segmentation (always 100)

#Variables for CNN architecture
CNN_FILTER_COUNT = 200 #number of filters used in convolutional layers (always 200)
KERNEL_SIZE = 8 #kernel / filter size of 8x8 (always 8)
STRIDE_LENGTH = 1 #stride for kernel application in convolutional layers (always 1)
DROPOUT_RATE = 0.5 #percentage of dropout in Dropout layer (always 0.5)
OUTPUT_CLASSES_COUNT = 10 #number of different target (gesture) classes (always 10)

#Variable for training, testing dataset split
PERCENT_TRAINING = 0.8 #Percentage of data that is used for training = 80% (always 0.8)

#Set names for targets (gesture labels)
LABELS = ["CLO","IND","LIT","MID","RIN","THI","THL","THM","THR","THU"]

#Define function for creating confusion matrix visualization
def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def main():

    #----Load Data----#
    # Read datafile into dataframe
    #USER NEEDS TO UPDATE THE FILENAME AND PATH/LOCATION IF DIFFERENT 
    f = 'compiled_data.csv' #compiled dataset filename
    path = '../data/' #update with path for locating compiled datset
    data = pd.read_csv(str(path+f), header = 0)
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


    callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
    opt = Adagrad(learning_rate=0.01)

    #----Create CNNs----#
    for cnn_index in range(NUM_SUBJECTS): #Only 1 CNN is created for a each subject
        EMG_CNN = Sequential(name="EMG_CNN"+str(cnn_index))
        EMG_CNN.add(Reshape((WINDOW_SIZE, INPUT_CHANN_COUNT), input_shape=(1600,)))
        EMG_CNN.add(Conv1D(filters=CNN_FILTER_COUNT, kernel_size=KERNEL_SIZE, activation='relu', strides=STRIDE_LENGTH, input_shape=(WINDOW_SIZE,INPUT_CHANN_COUNT)))  #Conv
        EMG_CNN.add(Conv1D(filters=CNN_FILTER_COUNT, kernel_size=KERNEL_SIZE, activation='relu', strides=STRIDE_LENGTH))                       #Conv
        EMG_CNN.add(MaxPool1D(pool_size=8, strides=8))                                                                                         #Max Pooling
        EMG_CNN.add(Conv1D(filters=CNN_FILTER_COUNT, kernel_size=KERNEL_SIZE, activation='relu', strides=STRIDE_LENGTH))                       #Conv
        EMG_CNN.add(Conv1D(filters=CNN_FILTER_COUNT, kernel_size=KERNEL_SIZE, activation='relu', strides=STRIDE_LENGTH))                       #Conv
        EMG_CNN.add(GlobalAveragePooling1D())                                                                                                  #Global Avg. Pooling
        EMG_CNN.add(Dropout(DROPOUT_RATE))                                                                                                     #Dropout
        EMG_CNN.add(Dense(OUTPUT_CLASSES_COUNT, activation='softmax'))                                                                            #Fully Connected
                                                                               #Fully Connected

        EMG_CNN.compile(loss='categorical_crossentropy',
                        optimizer=opt, 
                        metrics=['accuracy'])

        print(EMG_CNN.summary())

        #----Train CNN----#
        print("Training EMG_CNN on data...")
        history = EMG_CNN.fit(X_train[cnn_index], Y_train[cnn_index], batch_size=100, epochs=500, callbacks = [callback], validation_split=0.2, verbose=0)
        
	#----Visualization, Testing----#
        #Visualize training and validation curves, confusion matrix        
        print("\n--- Learning curve of model training ---\n")
        #Accuracy plot
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
        plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Epoch')
        plt.ylim(0)
        plt.legend()
    
        #Loss plot
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], "r--", label="Loss of training data")
        plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Training Epoch')
        plt.ylim(0)
        plt.legend()
        plt.show()

        #Evaluate the accuracy and loss value on test data
        print("\n--- Check against test data ---\n")
        score = EMG_CNN.evaluate(X_test[cnn_index], Y_test[cnn_index], verbose=1)
        print("\nAccuracy on test data: %0.2f" % score[1])
        print("\nLoss on test data: %0.2f" % score[0])

        #Display the confusion matrix for test data
        print("\n--- Confusion matrix for test data ---\n")
        y_pred_test = EMG_CNN.predict(X_test[cnn_index])
        # Take the class with the highest probability from the test predictions
        max_y_pred_test = np.argmax(y_pred_test, axis=1)
        max_y_test = np.argmax(Y_test[cnn_index], axis=1)
        show_confusion_matrix(max_y_test, max_y_pred_test)
 
        #Display classification report (which includes F1, precision, recall) for the test data
        print("\n--- Classification report for test data ---\n")
        print(classification_report(max_y_test, max_y_pred_test))
        


if __name__ == "__main__":
    main()