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

    elif architectureName == "Custom Model (1-Layer Network w 128 Filters)":
        imageSize = 128

        baseModel = Sequential()
        baseModel.add(Conv2D(128, (3, 3), activation='relu', input_shape=(128,128,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(9, activation='softmax'))

    elif architectureName == "Custom Model (1-Layer Network w 8 Filters)":
        imageSize = 128

        baseModel = Sequential()
        baseModel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(128,128,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(9, activation='softmax'))

    elif architectureName == "Custom Model (1-Layer Network w 4 Filters)":
        imageSize = 128

        baseModel = Sequential()
        baseModel.add(Conv2D(4, (3, 3), activation='relu', input_shape=(128,128,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(9, activation='softmax'))

    elif architectureName == "Custom Model (Lei - Linear)":
        imageSize = 128

        baseModel = Sequential()
        baseModel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(32, (3, 3), activation='relu'))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(64, (3, 3), activation='relu'))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(9, activation='softmax'))

    elif architectureName == "Custom Model (Lei CFCN)":
        imageSize = 128

        # Create CNN architecture
        modelInput = Input(shape = (128, 128, 3))

        padding_1 = ZeroPadding2D(padding = 1)(modelInput)
        conv2D_1a = Conv2D(32, (5, 5), strides = (2,2), activation='relu')(padding_1)
        conv2D_1b = Conv2D(32, (11, 11), strides = (4,4), activation='relu')(modelInput)
        # conv2D_1c = Conv2D(32, (5,5), activation='relu')(modelInput)
        conv2D_1 = Conv2D(32, (3,3), activation='relu')(modelInput)
        pool_1 = MaxPooling2D((2, 2), strides=(2, 2))(conv2D_1)

        merged1 = keras.layers.concatenate([pool_1, conv2D_1a])

        conv2D_2a = Conv2D(64, (5, 5), strides = (2,2), activation='relu')(merged1)
        conv2D_2 = Conv2D(64, (3,3), activation='relu')(merged1)
        # pool_2a = MaxPooling2D((4, 4), strides=(4,4))(conv2D_1c)
        pool_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2D_2)

        merged2 = keras.layers.concatenate([pool_2, conv2D_2a, conv2D_1b])

        padding_2 = ZeroPadding2D(padding = 1)(merged2)
        conv2D_3a = Conv2D(128, (5, 5), strides = (2,2), activation='relu')(padding_2)
        conv2D_3 = Conv2D(128, (3,3), activation='relu')(merged2)
        pool_3 = MaxPooling2D((2, 2), strides=(2, 2))(conv2D_3)

        merged3 = keras.layers.concatenate([pool_3, conv2D_3a])

        dense = Flatten()(merged3)
        output = Dense(128, activation='relu')(dense)
        dropout = Dropout(0.5)(output)
        predictions = Dense(9, activation='softmax')(dropout)
        baseModel = Model(inputs = modelInput, outputs = predictions)


    return imageSize, baseModel

def GetImages(datasetDirectory, imageSize, channels):

    if imageSize == 224:
        imagesNPYFile = "./LeiDatasetImages(224).npy"
        labelsNPYFile = "./LeiDatasetLabels(224).npy"
    elif imageSize == 299:
        imagesNPYFile = "./LeiDatasetImages(299).npy"
        labelsNPYFile = "./LeiDatasetLabels(299).npy"
    elif imageSize == 128:
        imagesNPYFile = "./LeiDatasetImages(128).npy"
        labelsNPYFile = "./LeiDatasetLabels(128).npy"

    if not os.path.isfile(imagesNPYFile):
        print("Images file not present, reading Images and Labels from raw dataset")
        images, labels = readLeiDataset.readTrainingData(datasetDirectory, imageSize, imageSize, channels)
        print("Shuffling images and labels")
        images, labels = shuffle(images, labels)

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

def EvaluateModel (history, modelName, saveDirectory):
    # Evaluate best model on test set
    if modelName == "MobileNetV2":
        with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
            modelFileName = os.path.join(saveDirectory, "Lei Dataset - " + modelName + ".h5")
            model = load_model(modelFileName)
    else:
        modelFileName = os.path.join(saveDirectory, "Lei Dataset - " + modelName + ".h5")
        model = load_model(modelFileName)

    # Save model scores
    print("Saving Model Scores")
    modelScoreFilePath = os.path.join(saveDirectory, modelName + ".txt")
    summaryCSVDirectory = os.path.join("./Results/", modelName + ".csv")
    with open(modelScoreFilePath, "w") as textFile:
        # Print Scores
        score = model.evaluate(x_test, y_test, verbose=0)
        predictions = model.predict(x_test)
        print("Overall loss on test set: {} \n Accuracy on test set: {}".format(score[0], score[1]), file=textFile)

    with open(summaryCSVDirectory, "a", newline="") as csvfile:
        scoreWriter = csv.writer(csvfile)
        scoreWriter.writerow([score[0], score[1]])

    # Save Predictions
    print("Saving Predictions to CSV file")
    predictionCSVfilePath = os.path.join(saveDirectory, modelName + ".csv")
    with open(predictionCSVfilePath, "w") as csvfile:
        predictionWriter = csv.writer(csvfile)
        for prediction in predictions:
            predictionWriter.writerow(prediction)

    # Save y_test
    print("Saving y_test")
    yTestFilePath = os.path.join(saveDirectory, modelName + " y_test.txt")
    with open(yTestFilePath, "w") as textFile:
        for yTest in y_test:
            print(yTest, file = textFile)

    # Save Model Summary
    print("Saving Model Summary")
    modelSummaryFilePath = os.path.join(saveDirectory, modelName + " Model Summary.txt")
    with open(modelSummaryFilePath, "w") as textFile:
            model.summary(print_fn=lambda x: textFile.write(x + '\n'))

    modelArchitecture = os.path.join(saveDirectory, 'LeiDataset ' + modelName + ' - Architecture.png')
    plot_model(model, to_file=modelArchitecture)

    accGraphFileName = os.path.join(saveDirectory, "Lei Dataset Model Accuracy" + "(" + modelName + ")" + ".png")
    lossGraphFileName = os.path.join(saveDirectory, "Lei Dataset Model Loss" + "(" + modelName + ")" + ".png")

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

    # Clear Session After Evaluating Model
    K.clear_session()

    return

if __name__ == "__main__":


    # Define Models to Include
    # models = ["Custom Model (1-Layer Network w 32 Filters)", "Custom Model (1-Layer Network w 64 Filters)", "Custom Model (1-Layer Network w 128 Filters)", "Custom Model (2-Layer Network)"]
    models = ["VGG16", "VGG19"]
    # models = ["Custom Model (1-Layer Network w 16 Filters)", "Custom Model (1-Layer Network w 8 Filters)", "Custom Model (1-Layer Network w 4 Filters)"]
    #models = ["Custom Model (1-Layer Network w 128 Filters)"]

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

            # Get training images and labels in categorical format
            # images, labels = GetImages(datasetDirectory, imageSize, channels)

            # Convert 1-dimensional class arrays to n-dimensional class matrices
            # labels = np_utils.to_categorical(labels - 1, classes)

            # Split the data into training and validation set
            # x_train, x_validation, y_train, y_validation = train_test_split(images, labels, test_size=0.3, random_state=np.random.randint(1000))

            # Split validation set further into testing set. Total = 70% training set, 15% validation, 15% testing set
            # x_validation, x_test, y_validation, y_test = train_test_split(x_validation, y_validation, test_size=0.5, random_state=np.random.randint(100))

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

