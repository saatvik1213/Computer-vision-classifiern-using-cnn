#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 07:04:02 2022

@author: saatvikchoudhary
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import cv2 as cv




class Model: 
    
    def __init__(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size= 3, activation= 'relu', input_shape= [64,64,3]))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))
        self.model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size= 3, activation= 'relu', ))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units =128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units =1, activation='sigmoid'))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        
    def train_model(self, class_num):
        
        self.train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        self.training_set = self.train_datagen.flow_from_directory(
                f'{class_num}',
                target_size=(64, 64),
                batch_size=32,
                class_mode= 'binary')
        self.model.fit(x = self.training_set, validation_data = self.test_set, epochs = 10)
        
    def predict(self, frame):
        frame = frame[1]
        cv.imwrite('frame.jpg',cv.cvtColor(cv.COLOR_RGB2GRAY))
        test_image = image.load_img('frame.jpg', target_size = (64, 64))
        test_image =image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0,)
        result = self.model.predict(test_image)
        self.training_set.class_indices
        return result
        
        
        
        








