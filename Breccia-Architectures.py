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

def DefineArchitecture(architectureName):
    if architectureName == "VGG16":
        imageSize = 224
        baseModel = VGG16(weights='imagenet', include_top=True)

    elif architectureName == "VGG19":
        imageSize = 224
        baseModel = VGG19(weights='imagenet', include_top=True)

    elif architectureName == "ResNet50":
        imageSize = 224
        baseModel = ResNet50(weights='imagenet', include_top=True)

    elif architectureName == "InceptionV3":
        imageSize = 299
        baseModel = InceptionV3(weights='imagenet', include_top=True)

    elif architectureName == "InceptionResNetV2":
        imageSize = 299
        baseModel = InceptionResNetV2(weights='imagenet', include_top=True)

    elif architectureName == "MobileNet":
        imageSize = 224
        baseModel = MobileNet(weights='imagenet', include_top=True)

    elif architectureName == "MobileNetV2":
        imageSize = 224
        baseModel = MobileNetV2(weights='imagenet', include_top=True)

    elif architectureName == "DenseNet121":
        imageSize = 224
        baseModel = DenseNet121(weights='imagenet', include_top=True)

    elif architectureName == "DenseNet169":
        imageSize = 224
        baseModel = DenseNet169(weights='imagenet', include_top=True)

    elif architectureName == "DenseNet201":
        imageSize = 224
        baseModel = DenseNet201(weights='imagenet', include_top=True)

    elif architectureName == "Custom Model (Breccia - Linear)":
        imageSize = 256

        baseModel = Sequential()
        baseModel.add(Conv2D(128, (3, 3), activation='relu', input_shape=(imageSize, imageSize, 3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(64, (3, 3), activation='relu'))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(32, (3, 3), activation='relu'))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(32, (3, 3), activation='relu'))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(32, (3, 3), activation='relu'))
        baseModel.add(Flatten())
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        baseModel.add(Dense(2, activation='softmax'))

    elif architectureName == "Custom Model (Breccia - CFNC)":
        imageSize = 256

        # Create CNN architecture
        modelInput = Input(shape = (256, 256, 3))

        padding_1 = ZeroPadding2D(padding = 1)(modelInput)

        conv2D_1a = Conv2D(128, (3,3), activation='relu')(modelInput)
        conv2d_1b = Conv2D(128, (5,5), strides=(2, 2), activation='relu')(padding_1)
        conv2d_1c = Conv2D(128, (11,11), strides=(4,4), activation='relu')(modelInput)

        pool_1 = MaxPooling2D((2, 2), strides=(2, 2))(conv2D_1a)

        merged_1 = keras.layers.concatenate([pool_1, conv2d_1b])

        conv2D_2a = Conv2D(64, (3,3), activation='relu')(merged_1)
        conv2D_2b = Conv2D(64, (5, 5), strides=(2,2), activation='relu')(merged_1)

        pool_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2D_2a)

        merged_2 = keras.layers.concatenate([pool_2, conv2D_2b, conv2d_1c])

        pool_2a = MaxPooling2D((2,2), strides=(2,2))(merged_2)

        padding_2 = ZeroPadding2D(padding = 1)(merged_2)

        conv2D_3a = Conv2D(32, (3,3), activation='relu')(merged_2)
        conv2D_3b = Conv2D(32, (5,5), strides=(2,2), activation='relu')(padding_2)
        conv2D_2c = Conv2D(64, (11,11), strides=(4,4), activation='relu')(padding_2)

        pool_3 = MaxPooling2D((2, 2), strides=(2, 2))(conv2D_3a)

        merged_3 = keras.layers.concatenate([pool_3, conv2D_3b])

        padding_3 = ZeroPadding2D(padding = 1)(merged_3)

        conv2D_4a = Conv2D(32, (3,3), activation='relu')(merged_3)
        conv2D_4b = Conv2D(32, (5,5), strides=(2,2), activation='relu')(padding_3)
        conv2D_4c = Conv2D(32, (5,5), strides=(2,2), activation='relu')(pool_2a)
        conv2D_4d = Conv2D(32, (7,7), strides=(2,2), activation='relu')(merged_3)

        pool_4 = MaxPooling2D((2, 2), strides=(2, 2))(conv2D_4a)

        merged_4 = keras.layers.concatenate([pool_4, conv2D_4b, conv2D_2c, conv2D_4c])

        conv2D_5 = Conv2D(32, (3,3), activation='relu')(merged_4)

        merged_5 = keras.layers.concatenate([conv2D_5, conv2D_4d])

        flatten = Flatten()(merged_5)
        dropout = Dropout(0.5)(flatten)
        dense_1 = Dense(128, activation='relu')(dropout)
        dropout = Dropout(0.5)(dense_1)
        dense_1 = Dense(64, activation='relu')(dropout)
        predictions = Dense(2, activation='softmax')(dropout)

        baseModel = Model(inputs = modelInput, outputs = predictions)

    return imageSize, baseModel

def GetImages(datasetDirectory, imageSize, channels):

    if imageSize == 224:
        imagesNPYFile = "./BrecciaDatasetImages(224).npy"
        labelsNPYFile = "./BrecciaDatasetLabels(224).npy"
    elif imageSize == 299:
        imagesNPYFile = "./BrecciaDatasetImages(299).npy"
        labelsNPYFile = "./BrecciaDatasetLabels(299).npy"
    elif imageSize == 256:
        imagesNPYFile = "./BrecciaDatasetImages(256).npy"
        labelsNPYFile = "./BrecciaDatasetLabels(256).npy"

    if not os.path.isfile(imagesNPYFile):
        print("Images file not present, reading Images and Labels from raw dataset")
        images, labels = readBrecciaDataset.readTrainingData(datasetDirectory, imageSize, imageSize, channels)

    else:
        print("Images file present! Loading Images and Labels...")
        images = np.load(imagesNPYFile)
        labels = np.load(labelsNPYFile)
        print("{} images and {} classes".format(len(images), len(labels)))
        print("Shuffling images and labels")
        images, labels = shuffle(images, labels)

    return images, labels

def relu6(x):
	return K.relu(x, max_value=6)

def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def EvaluateModel (history, modelName, saveDirectory):
    # Evaluate best model on test set
    if modelName == "MobileNetV2":
        with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
            modelFileName = os.path.join(saveDirectory, "Breccia Dataset - " + modelName + ".h5")
            model = load_model(modelFileName, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    else:
        modelFileName = os.path.join(saveDirectory, "Breccia Dataset - " + modelName + ".h5")
        model = load_model(modelFileName, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})

    # Save model scores
    print("Saving Model Scores")
    modelScoreFilePath = os.path.join(saveDirectory, modelName + ".txt")
    with open(modelScoreFilePath, "w") as textFile:
        # Print Scores
        score = model.evaluate(x_test, y_test, verbose=0)
        predictions = model.predict(x_test)
        print("Overall loss on test set:{}, Accuracy on test set:{}, Precision: {}, Recall: {}, F-score: {}".format(score[0], score[1], score[2], score[3], score[4]), file=textFile)

    # Save Predictions
    print("Saving Predictions")
    predictionCSVfilePath = os.path.join(saveDirectory, modelName + ".csv")
    with open(predictionCSVfilePath, "w") as csvfile:
        predictionWriter = csv.writer(csvfile)
        for prediction in predictions:
            predictionWriter.writerow(prediction)

    # Save y_test
    print("Saving y_test")
    yTestFilePath = os.path.join(saveDirectory, modelName + " y_test.txt")
    with open(yTestFilePath, "w") as textFile:
        print(y_test, file = textFile)

    # Save Model Summary
    print("Saving Model Summary")
    modelSummaryFilePath = os.path.join(saveDirectory, modelName + " Model Summary.txt")
    with open(modelSummaryFilePath, "w") as textFile:
            model.summary(print_fn=lambda x: textFile.write(x + '\n'))

    modelArchitecture = os.path.join(saveDirectory, 'BrecciaDataset ' + modelName + ' - Architecture.png')
    plot_model(model, to_file=modelArchitecture)

    accGraphFileName = os.path.join(saveDirectory, "Breccia Dataset Model Accuracy" + "(" + modelName + ")" + ".png")
    lossGraphFileName = os.path.join(saveDirectory, "Breccia Dataset Model Loss" + "(" + modelName + ")" + ".png")
    precisionGraphFileName = os.path.join(saveDirectory, "Breccia Dataset Model Precision" + "(" + modelName + ")" + ".png")
    recallGraphFileName = os.path.join(saveDirectory, "Breccia Dataset Model Recall" + "(" + modelName + ")" + ".png")
    fScoreGraphFileName = os.path.join(saveDirectory, "Breccia Dataset Model Fscore" + "(" + modelName + ")" + ".png")

    # Plot accuracy history
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(modelName + ' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='center right')
    plt.savefig(accGraphFileName)
    plt.close()

    # Plot loss history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(modelName + 'Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='center right')
    plt.savefig(lossGraphFileName)
    plt.close()

    # Plot precision history
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title(modelName + ' Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='center right')
    plt.savefig(precisionGraphFileName)
    plt.close()

    # Plot recall history
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title(modelName + ' Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='center right')
    plt.savefig(recallGraphFileName)
    plt.close()

    # Plot f-score history
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title(modelName + ' F-Score')
    plt.ylabel('F-Score')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='center right')
    plt.savefig(fScoreGraphFileName)
    plt.close()

    # Clear Session After Evaluating Model
    K.clear_session()
    return

if __name__ == "__main__":

    # Define Models to Include
    models = ["DenseNet201", "VGG16", "VGG19", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2", "DenseNet121", "DenseNet169"]
    # Define hyper parameters
    channels = 3
    classes = 2
    trials = 9
    epochs = 200
    batch_size = 16
    datasetDirectory = './Breccia Dataset'

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
            saveDirectory = "./BrecciaResults/" + model  + " Trial " + str(i)
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