# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:40:41 2018

@author: Alexis Pascual
"""


import os
import os.path
import keras
import readBrecciaDataset
import numpy as np
import matplotlib.pyplot as plt
import csv

from keras.models import Sequential, load_model, Model

# Generic NN layers
from keras.layers import Dense, Dropout, Flatten

# Generic CNN layers
from keras.layers import Conv2D, MaxPooling2D, Input, DepthwiseConv2D, ZeroPadding2D

# Import utils
from keras.utils import np_utils, plot_model
from sklearn.utils import shuffle

# Import Callbacks
from keras.callbacks import ModelCheckpoint

# Import Image generator
from keras.preprocessing.image import ImageDataGenerator

# Import Dataset Splitter
from sklearn.model_selection import train_test_split

# Import backend
from keras import backend as K

#Import optimizer
from keras import optimizers

#Import Regularizer
from keras import regularizers

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

# Import tools
from tools.models import DefineArchitecture, EvaluateModel
from tools.readImages import ReadTrainingData, GetImages

if __name__ == "__main__":

    config_file = "./config/config.yaml"

    if not os.path.exists(config_file):
        print("Config file not found!")
        exit(1)
    else:
        with open(config_file, 'r') as f:
            args = yaml.safe_load(f)

    # Force number of classes to 2
    classes = 2

    models = args['models']
    channels = args['channels']
    trials = args['trials']
    epochs = args['epochs']
    batch_size = args['batch_size']
    dataset = args['dataset']
    datasetDirectory = args['datasetDirectory']

    for i in range(trials):
        for model in models:

            print("Training in model " + model)
            # Get image size requirement for each base model
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

            adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

            # Compile model
            modifiedModel.compile(loss='categorical_crossentropy',
                                  optimizer=adam,
                                  metrics=['accuracy', precision, recall, f1])

            print(modifiedModel.summary())

            # Get training images
            images, labels = GetImages(datasetDirectory, imageSize, channels)

            # Convert 1-dimensional class arrays to n-dimensional class matrices
            labels = np_utils.to_categorical(labels - 1, classes)

            # Split the data into training and validation set
            x_train, x_validation, y_train, y_validation = train_test_split(images, labels, test_size=0.3, random_state=np.random.randint(1000))

            # Split validation set further into testing set. Total = 70% training set, 15% validation, 15% testing set
            x_validation, x_test, y_validation, y_test = train_test_split(x_validation, y_validation, test_size=0.5, random_state=np.random.randint(100))

            # this is the augmentation configuration we will use for training
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode="wrap",
                shear_range=0.2,
                zoom_range=0.2)

            # Define model file name
            saveDirectory = "./" + dataset + " Results/" + model  + " Trial " + str(i)
            modelFilePath = os.path.join(saveDirectory, "Breccia Dataset - " + model + ".h5")

            if not  os.path.exists(saveDirectory):
                os.makedirs(saveDirectory)
            else:
                print("Save Directory Exists!")

            # Create a checkpoint to save best model
            checkpointer = ModelCheckpoint(filepath=modelFilePath, verbose=1, save_best_only=True)

            # Compute quantities required for feature-wise normalization
            datagen.fit(x_train)

            # fits the model on batches with real-time data augmentation and save the model only if validation loss decreased
            try:
                history = modifiedModel.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                steps_per_epoch=len(x_train) / batch_size,
                                epochs=epochs,
                                validation_data=(x_validation, y_validation),
                                callbacks=[checkpointer])
            except KeyboardInterrupt:
                print("Clearing session. Hopefully this works.")
                K.clear_session()

            EvaluateModel(history, model, saveDirectory)