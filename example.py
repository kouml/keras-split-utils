import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import numpy as np

import split_utils

original_dir = './data'
shapes = (28, 28)
batch_size = 32
validation_split = 0.1

# all data in train_dir which are alias to original_data.(both dir is temporary directory)
# don't clear base_dir, because this directory holds on temp directory.
base_dir, train_dir, val_dir = split_utils.train_valid_split(original_dir, validation_split, seed=1)

# generator for train data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=shapes,
    batch_size=batch_size,
    color_mode='grayscale'
)

# generator for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=shapes,
    batch_size=batch_size,
    color_mode='grayscale'
)

print('the ratio of validation_split is {}'.format(validation_split))
print('the size of train_dir is {}'.format(train_gen.n))
print('the size of val_dir is {}'.format(val_gen.n))

def build_dense_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.summary()
    return model

model = build_dense_model()
train_steps = np.ceil(train_gen.n / batch_size)
valid_steps = np.ceil(val_gen.n / batch_size)

hist = model.fit(train_gen,
                 epochs=10,
                 validation_data=val_gen,
                 steps_per_epoch=train_steps,
                 validation_steps=valid_steps
)
