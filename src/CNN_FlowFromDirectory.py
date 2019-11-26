# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:19:34 2018

@author: Alexis Pascual
"""

import os
import os.path
import readLeiDataset
import keras
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential, load_model, Model
from keras.utils import plot_model

# Generic NN layers
from keras.layers import Dense, Dropout, Flatten

# Generic CNN layers
from keras.layers import Conv2D, MaxPooling2D, Input, DepthwiseConv2D, ZeroPadding2D

# Import utils
from keras.utils import np_utils
from sklearn.utils import shuffle

# Import Callbacks
from keras.callbacks import ModelCheckpoint

# Import Image generator
from keras.preprocessing.image import ImageDataGenerator

# Import Dataset Splitter
from sklearn.model_selection import train_test_split

# Import backend
from keras import backend as K

# Import Models
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.utils.generic_utils import CustomObjectScope

if __name__ == "__main__":


    # Define Models to Include
    models = ["Custom Model (1-Layer Network w 32 Filters)", "Custom Model (1-Layer Network w 64 Filters)", "Custom Model (1-Layer Network w 128 Filters)", "Custom Model (2-Layer Network)"]

    # Define hyper parameters
    channels = 3
    classes = 5
    trials = 9
    epochs = 200
    batch_size = 16
    datasetDirectory = './Rock Dataset/training_set'

    for i in range(trials):
        for model in models:

            print("Training in model " + model)
            # Get image size requirement for each base model
            baseModel = None
            modifiedModel = None
            selfTestImages = None
            selfTestLabels = None
            images = None
            labels= None
            imageSize, baseModel = DefineArchitecture(model)
            checkModelName = model.split(" ")
            checkModelName = checkModelName[0]

            if not checkModelName == "Custom":
                # Define input and input shape
                input = Input(shape=(imageSize,imageSize, channels),name = 'image_input')

                # Define Output Layer
                x = Dense(classes, activation='softmax', name='predictions')(baseModel.layers[-2].output)

                # Tie it all together
                modifiedModel = Model(baseModel.input, output = x)

                # Set all layers to be non trainable except for the last layer
                for layer in modifiedModel.layers[:(len(modifiedModel.layers) - 1)]:
                    layer.trainable = False

            elif checkModelName == "Custom":
                modifiedModel = baseModel

            # Define Adam optimizer
            adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

            # Compile model
            modifiedModel.compile(loss='categorical_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'])

            print(modifiedModel.summary())

            # this is the augmentation configuration we will use for training
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode="wrap",
                shear_range=0.2,
                zoom_range=0.2,
                validation_split=0.3)

            # Define Save Directory
            saveDirectory = "./Results/" + model + " Trial " + str(i)

            # Define model file name
            modelFilePath = os.path.join(saveDirectory, "Lei Dataset - " + model + ".h5")

            if not  os.path.exists(saveDirectory):
                print("Creating Save Directory")
                os.makedirs(saveDirectory)
            else:
                print("Save Directory Exists!")

            # Create a checkpoint to save best model
            checkpointer = ModelCheckpoint(filepath=modelFilePath, verbose=1, save_best_only=True)

            # Compute quantities required for feature-wise normalization
            # datagen.fit(x_train)

            trainGenerator = datagen.flow_from_directory(
                datasetDirectory,
                target_size = (imageSize, imageSize),
                batch_size = batch_size,
                class_mode = 'categorical',
                subset = 'training'
                )

            validationGenerator = datagen.flow_from_directory(
                datasetDirectory,
                target_size = (imageSize, imageSize),
                batch_size = batch_size,
                class_mode = 'categorical',
                subset = 'validation'
                )

            step_size_train=trainGenerator.n//trainGenerator.batch_size
            step_size_validation = validationGenerator.n//validationGenerator.batch_size

            # fits the model on batches with real-time data augmentation and save the model only if validation loss decreased
            try:
                history = modifiedModel.fit_generator(trainGenerator,
                                steps_per_epoch=step_size_train,
                                epochs=epochs,
                                validation_data=validationGenerator,
                                validation_steps=step_size_validation
                                callbacks=[checkpointer])
                
            except KeyboardInterrupt:
                print("Clearing session. Hopefully this works.")
                K.clear_session()

            # Evaluate Model
            EvaluateModel(history, model, saveDirectory)

