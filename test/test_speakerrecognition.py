"""
PoC using Tensorflow + Keras
"""
import glob
import numpy as np
import random
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

SEED = 2017
DATA_DIR = 'Data/'

files = glob.glob(DATA_DIR + "*.wav")
X_train, X_val = train_test_split(files, test_size=0.2, random_state=SEED)

print('# Training examples: {}'.format(len(X_train)))
print('# Validation examples: {}'.format(len(X_val)))

labels = []
for i in range(len(X_train)):
    label = X_train[i].split('/')[-1].split('_')[1]
    if label not in labels:
        labels.append(label)
print(labels)

label_binarizer = LabelBinarizer()
label_binarizer.fit(list(set(labels)))

def one_hot_encode(x): return label_binarizer.transform(x)

n_features = 20
max_length = 80
n_classes = len(labels)

def batch_generator(data, batch_size=16):
    while 1:
        random.shuffle(data)
        X, y = [], []
        for i in range(batch_size):
            wav = data[i]
            wave, sr = librosa.load(wav, mono=True)
            label = wav.split('/')[-1].split('_')[1]
            y.append(one_hot_encode(label))
            mfcc = librosa.feature.mfcc(wave, sr)
            mfcc = np.pad(mfcc, ((0,0), (0, max_length-
            len(mfcc[0]))), mode='constant', constant_values=0)
            X.append(np.array(mfcc))
        yield np.array(X), np.array(y)

learning_rate = 0.001
batch_size = 64
n_epochs = 50
dropout = 0.5

input_shape = (n_features, max_length)
steps_per_epoch = 50

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=input_shape, dropout=dropout))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(n_classes, activation='softmax'))

opt = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model.summary()

callbacks = [ModelCheckpoint('checkpoints/voice_recognition_best_model_{epoch:02d}.hdf5', save_best_only=True),
            EarlyStopping(monitor='val_acc', patience=2)]

history = model.fit_generator(
    generator=batch_generator(X_train, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    verbose=1,
    validation_data=batch_generator(X_val, 32),
    validation_steps=5,
    callbacks=callbacks
)

