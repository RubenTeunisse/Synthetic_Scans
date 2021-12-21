""" Created by Ruben Teunisse 2021 """

import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import *
from keras import backend as K
from keras.models import Model

from data import get_1_UCL, get_CERMEP
from utils import tune_threshold, show_results


#######################
### Hyperparameters ###
#######################

nr_feats = 32
nr_subjects = 37
nr_epochs = 400
batch_size = 150
threshold = 2.75
trim = 140

CERMEP = True
new_preprocessing = True



################
### Get Data ###
################

if CERMEP:
    if new_preprocessing:
        data = get_CERMEP(path="D:\\Human Skulls\\CERMEP-IDB-MRXFDG",
                          nr_subjects=nr_subjects,
                          threshold=threshold,
                          trim=trim,
                          ds_factor=2)
        joblib.dump(data, f'./CERMEP_{nr_subjects}_th_{threshold}_trim_{trim}')
    else:
        data = joblib.load(f'./CERMEP_{nr_subjects}_th_{threshold}_trim_{trim}')
else:
    data = get_1_UCL(path="D:\\Human Skulls\\UCL\\UCL\\0698168\\x1_7_Central_Tremor_Std.nii.gz", ds_factor=2)

x_train, x_test, y_train, y_test, input_shape = data
img_rows, img_cols = input_shape[:-1]



###############
###  MODEL  ###
###############

### Encoder ###
print("Build model")
encoder_input = Input(shape=input_shape)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder_input)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(64, activation='linear')(x)
x = Dropout(0.5)(x)
encoder_output = Dense(nr_feats, activation='linear')(x)

### Decoder ###
direct_input = Input(shape=encoder_output.shape)
decoder_start = Dense(64, activation='linear')(direct_input)
x = Dense(64, activation='linear')(decoder_start)
x = Dropout(0.5)(x)
x = Dense(128, activation='linear')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='linear')(x)
x = Dropout(0.5)(x)
decoder_output = Dense(img_rows*img_cols, activation='sigmoid')(x)

# Get models
encoder = Model(encoder_input, encoder_output)
decoder = Model(direct_input, decoder_output)
model = Model(encoder_input, decoder(encoder_output))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.nadam(), metrics=['accuracy'])
print(model.summary())

# Train model
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=nr_epochs,
          verbose=1,
          validation_data=(x_test, y_test))



###############
### RESULTS ###
###############

# Evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['val_loss'])
plt.title("validation loss")
plt.show()

# Store Models
joblib.dump(model, './model_high_res')
joblib.dump(encoder, './encoder_high_res')
joblib.dump(decoder, './decoder_high_res')

# Generate intermediate output
show_results(model, encoder, x_train, x_test, trim, img_rows, img_cols)
