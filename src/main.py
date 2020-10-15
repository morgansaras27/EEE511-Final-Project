#main for for ANC Final Project.
#Creates, trains, and calls a CNN using the tensorflow library.

#This CNN is based off the research/work discussed in the following paper:


#Imports
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense



def main():

    #----Load Data----#


    #----Data Preprocessing----#


    #----Create CNN----#
    EMG_CNN = Sequential(name="EMG_CNN")

    EMG_CNN.add(Dense(10, activation='relu', use_bias=True,input_shape=(5,)))
    EMG_CNN.add(Dense(10, activation='relu', use_bias=True))
    
    EMG_CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(EMG_CNN.summary())

    #----Train CNN----#
    print("Training EMG_CNN on data...NO ACTUAL DATA")
    #history = EMG_CNN.fit()

    #----Run CNN----#



if __name__ == "__main__":
    main()