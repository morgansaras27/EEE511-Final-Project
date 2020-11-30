# EMG signal gesture classification using Deep Learning
This project is intended to intake raw EMG signals and output classification of one of ten (10) hand gestures. 

## Dataset(s)
### Publicly Available Dataset
The publicly available dataset that may be utilized with this implementation can be found at: 
- EMG Repository (select dataset under 1-DETECTING INDIVIDUAL AND COMBINED FINGERS MOVEMENTS): https://www.rami-khushaba.com/electromyogram-emg-repository.html 

Note: This EMG Repository will take you to the following OneDrive Folder with the data: https://onedrive.live.com/?authkey=%21As%5FiNPKzgU6LJCU&id=AAA78954F15E6559%21295&cid=AAA78954F15E6559

This dataset is provided in individual files for each subject and each gesture. However, for use with the model, these individual files must be properly compiled and formatted. 
To do this, you may utilize our code in generate_dataset.py within the "data" folder.

### Self-collected Dataset
A subset of our self-collected data are also available in the "data" folder. These data have already been compiled and formatted for use with the model.


## Installation
The virtual environment must be installed in order to ensure the proper libraries, pacakages, and versions are utilized to run the model. To do this, follow the below instructions:
1. Install conda
```bash
pip install conda
```
2. Create a new virtual environment using the environment.yml file located in our GitHub
```bash
conda env create -f environment.yml
```
3. Activate the virual environment (named ANC_FinalProject)
```bash
conda activate ANC_FinalProject
```
Note: the virtual environment needs only to be installed and created once, but must be activated prior to each session of utilizing the code. 


## Usage
Once the python files are downloaded and virtual environment created and active, open main.py and ensure that the filename and path are appropriate for where and how you have saved your dataset. To segment and preprocess the data as well as train and test the model, run main.py.
 


## Google Colab Implementation
A Google Colab implementation of this code is available at: https://colab.research.google.com/drive/1fxV3S6-BJthJyyYklbjk9D5Um9VbmnQR?usp=sharing. 
When running with this implementation follow these steps:
1. Upload desired dataset file to your own personal Google Drive.
2. Update the Google Colab implementation to include the proper file path and filename.
3. Starting at the top, run each block of code in sequence. 