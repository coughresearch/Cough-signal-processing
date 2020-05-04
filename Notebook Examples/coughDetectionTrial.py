# Import.......

import librosa
import os
import itertools
import glob
from sksound.sounds import Sound
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from librosa.feature import melspectrogram
from librosa.util import normalize
from librosa.display import waveplot
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dense, Activation, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

# Functions to Extract Features.................

#Read original data

def readCoughData(file):
    origData,origSampFreq = librosa.load(file, sr=None)
    return origData, origSampFreq

# resample original data to 16000 Khz

def resample(originalData, origSampFreq, targetSampFreq):
    resampledData = librosa.resample(originalData, origSampFreq, targetSampFreq)
    return resampledData

# Normalize Sound Data

def normalizeSound(resampledData, axis):
    """ Axis is 0 for row-wise and 1
    for column wise"""
    normalizedData = normalize(resampledData, axis)
    return normalizedData

# Calculate Mel-Spectogram

def calculateMelSpectogram(normalizedData, hop_length, win_length, sr):
    #newSamplingFreq = 16000
    S=librosa.feature.melspectrogram(normalizedData, sr=sr, hop_length=hop_length, win_length=win_length)
    return S

# plot orginal time domain data

def plotSound(soundData, sr, x_axis_string):
    waveplot(soundData, sr, x_axis=x_axis_string)

#Plot melspectogram

def plotMelSpectogram(S, sr, ref=np.max):
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=16000,)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

# Function to Extract MelSpectrum
def featureExtraction(audioFile, targetSampFreq, axis, hop_length,win_length):
    y, y_sr = readCoughData(file=audioFile)
    resampledData = resample(originalData=y, origSampFreq=y_sr, targetSampFreq=targetSampFreq)
    normalizedData = normalizeSound(resampledData, axis=axis)
    S = calculateMelSpectogram(normalizedData=normalizedData, hop_length=hop_length, win_length=win_length, sr=targetSampFreq)
    #plotSound(soundData=normalizedData, sr=targetSampFreq,x_axis_string='time')
    #plotMelSpectogram(S, sr=targetSampFreq, ref=np.max)
    return S

# Training ...............


# Creating directory and the required arrays .........
trainingDataDirectory = 'C:\\Users\\tpaulraj\\Dropbox\\Projects\\Cough Research\\Cough-signal-processing\\Notebook Examples\\Data\\Training Data'
dirList = os.listdir(trainingDataDirectory)
soundfiles=[]
featureArray = np.zeros((200,128,313)) #array to store features
classArray = np.zeros((200,1)) #array to store classes

# Globbing all the sound files........
for dir in dirList:
    currentDirFile = glob.glob(trainingDataDirectory+'\\'+dir+'\\'+'*.ogg')
    soundfiles.append(currentDirFile)

# Flattenning soundfiles list into a single list...........
listOfAllSoundFiles=list(itertools.chain(*soundfiles))
shuffle(listOfAllSoundFiles)

#creating classes labels in the classArray......
for i in range(len(listOfAllSoundFiles)):
    if listOfAllSoundFiles[i].split('\\')[-2] == 'Coughing':
        classArray[i] = 1
    else:
        classArray[i] = 0

#OneHotEncoding classes

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(classArray)
y_train = enc.transform(classArray).toarray()

# Extracting features for all the 200 files and collecting them in featureArray
for num in range(len(listOfAllSoundFiles)):
    print('processing \n' + listOfAllSoundFiles[num])
    oneFileFeature = featureExtraction(listOfAllSoundFiles[num], targetSampFreq=16000, axis=0, hop_length=256, win_length=512)
    if oneFileFeature.shape[1] >313:
        reshapedOneFileFeature = oneFileFeature[:,:313]
        featureArray[num] = reshapedOneFileFeature
    else:
        featureArray[num] = oneFileFeature



# Cough detection model...........
def cough_detection_model():
    input_layer = Input((128,313,1))
    x = MaxPooling2D(pool_size=(2, 2))(input_layer)
    x = Conv2D(filters=32,kernel_size=(5,5),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=32,kernel_size=(5,5),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(input_layer)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(2,activation = 'softmax')(x)
    model = Model(inputs=input_layer,outputs=output_layer)
    adam = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model

X_train = featureArray                         # load your data here shape (200,128,313)
X_train = X_train.reshape((200,128,313,1))
#y_train = classArray                        # load your labels here shape (80,1)
#y_train = classArray.reshape((200,1,1))                     # one_hot_encoding
number_of_epochs = 50 # number of times you fed each data on X_train to the model
#model = cough_detection_model() # here you have to call the model you want to use, in this case DL_MC
model = cough_detection_model()
print('# Fit model on training data')
history = model.fit(X_train, y_train,
                    batch_size = 4,
                    epochs = number_of_epochs, validation_data = (X_train,y_train)) #I have set same data for training and
                                                # for validation because we have few instances, later when we have
                                                #more data we will make an split train/validation/test
print('\nhistory dict:', history.history)