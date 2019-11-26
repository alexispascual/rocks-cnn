import os
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

def DefineArchitecture(architectureName, classes, customImageSize):
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

    elif architectureName == "Custom Model (1-Layer Network w 32 Filters)":
        imageSize = customImageSize

        baseModel = Sequential()
        baseModel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(customImageSize,customImageSize,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(classes, activation='softmax'))

    elif architectureName == "Custom Model (2-Layer Network)":
        imageSize = customImageSize

        baseModel = Sequential()
        baseModel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(customImageSize,customImageSize,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(32, (3, 3), activation='relu'))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(classes, activation='softmax'))

    elif architectureName == "Custom Model (1-Layer Network w 64 Filters)":
        imageSize = customImageSize

        baseModel = Sequential()
        baseModel.add(Conv2D(64, (3, 3), activation='relu', input_shape=(customImageSize,customImageSize,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(classes, activation='softmax'))

    elif architectureName == "Custom Model (1-Layer Network w 128 Filters)":
        imageSize = customImageSize

        baseModel = Sequential()
        baseModel.add(Conv2D(customImageSize, (3, 3), activation='relu', input_shape=(customImageSize,customImageSize,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(classes, activation='softmax'))

    elif architectureName == "Custom Model (3-Layers)":
        imageSize = customImageSize

        baseModel = Sequential()
        baseModel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(customImageSize,customImageSize,3)))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(32, (3, 3), activation='relu'))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Conv2D(64, (3, 3), activation='relu'))
        baseModel.add(MaxPooling2D(pool_size=(2,2)))
        baseModel.add(Flatten())
        baseModel.add(Dense(128, activation='relu'))
        baseModel.add(Dropout(0.5))
        baseModel.add(Dense(classes, activation='softmax'))

    elif architectureName == "Custom Model (CFCN-Shallow)":
        imageSize = customImageSize

        # Create CNN architecture
        modelInput = Input(shape = (customImageSize, customImageSize, 3))

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
        predictions = Dense(classes, activation='softmax')(dropout)
        baseModel = Model(inputs = modelInput, outputs = predictions)

    elif architectureName == "Custom Model (Linear-5 layers)":
        imageSize = customImageSize

        baseModel = Sequential()
        baseModel.add(Conv2D(128, (3, 3), activation='relu', input_shape=(customImageSize, customImageSize, 3)))
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
        baseModel.add(Dense(classes, activation='softmax'))

    elif architectureName == "Custom Model (CFNC-Deep)":
        imageSize = customImageSize

        # Create CNN architecture
        modelInput = Input(shape = (customImageSize, customImageSize, 3))

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
        predictions = Dense(classes, activation='softmax')(dropout)

        baseModel = Model(inputs = modelInput, outputs = predictions)

    return imageSize, baseModel

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

def EvaluateModel (dataset, history, modelName, saveDirectory):
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

    modelArchitecture = os.path.join(saveDirectory, dataset + ' - ' + modelName + ' - Architecture.png')
    plot_model(model, to_file=modelArchitecture)

    accGraphFileName = os.path.join(saveDirectory, dataset + " Model Accuracy" + "(" + modelName + ")" + ".png")
    lossGraphFileName = os.path.join(saveDirectory, dataset + " Model Loss" + "(" + modelName + ")" + ".png")
    precisionGraphFileName = os.path.join(saveDirectory, dataset + " Model Precision" + "(" + modelName + ")" + ".png")
    recallGraphFileName = os.path.join(saveDirectory, dataset + " Model Recall" + "(" + modelName + ")" + ".png")
    fScoreGraphFileName = os.path.join(saveDirectory, dataset + " Model F-Score" + "(" + modelName + ")" + ".png")

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
    if history.history['precision']:
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title(modelName + ' Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='center right')
        plt.savefig(precisionGraphFileName)
        plt.close()

    # Plot recall history
    if history.history['recall']:
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title(modelName + ' Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='center right')
        plt.savefig(recallGraphFileName)
        plt.close()

    # Plot f-score history
    if history.history['f1']:
        plt.plot(history.history['f1'])
        plt.plot(history.history['val_f1'])
        plt.title(modelName + ' F-Score')
        plt.ylabel('F-Score')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='center right')
        plt.savefig(fScoreGraphFileName)
        plt.close()

    return