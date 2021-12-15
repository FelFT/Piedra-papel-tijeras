import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import pathlib

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


def modelo_CNN():
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape=(180, 180, 3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

data_dir = os.getcwd()+'/imgs'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))

mano_dict = {
    'piedra' : list(data_dir.glob('piedra/*')), 
    'papel' : list(data_dir.glob('papel/*')),
    'tijeras': list(data_dir.glob('tijeras/*')), 
    'arriba' : list(data_dir.glob('arriba/*')),
    'abajo': list(data_dir.glob('abajo/*')),
}

mano_labels_dict = {
    'piedra': 0,
    'papel': 1,
    'tijeras': 2,
    'arriba': 3,
    'abajo': 4,
    }

X, y = [], []

for mano_name, images in mano_dict.items():
    print(mano_name)
   
    for image in images:
        img = cv2.imread(str(image))
        #resized_img = cv2.resize(img,(180,180))
        X.append(img) 
        y.append(mano_labels_dict[mano_name])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

y_train = np_utils.to_categorical( y_train )
y_test = np_utils.to_categorical( y_test )

num_classes = y_test.shape[1]
num_classes


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

datagen = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1, 
                         shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, 
                         fill_mode="nearest")

model =modelo_CNN()

history1 = model.fit_generator(datagen.flow(X_train_scaled, y_train, batch_size=32),
	validation_data=(X_test_scaled, y_test), steps_per_epoch=len(X_train_scaled) // 32,
	epochs=10, verbose=1)

model.save('model.h5')

