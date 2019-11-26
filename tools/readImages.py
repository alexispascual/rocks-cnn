import cv2
import numpy as np

from sklearn.utils import shuffle

def GetImages(dataset, datasetDirectory, imageSize, channels):

    imagesNPYFile = "./" + dataset + "_Images" + "("+ str(imageSize) + ").npy"
    labelsNPYFile = "./" + dataset + "_Labels" + "("+ str(imageSize) + ").npy"

    if not os.path.isfile(imagesNPYFile):
        print("Images file not present, reading Images and Labels from raw dataset")
        images, labels = readTrainingData(dataset, datasetDirectory, imageSize, imageSize, channels)

    else:
        print("Images file present! Loading Images and Labels...")
        images = np.load(imagesNPYFile)
        labels = np.load(labelsNPYFile)
        print("{} images and {} classes".format(len(images), len(labels)))
        print("Shuffling images and labels")
        images, labels = shuffle(images, labels)

    return images, labels

def ReadTrainingData(dataset, directory, imageLength, imageWidth, channels):

    print('Reading Dataset from {}'.format(directory))
    images = []
    labels = []
    currentClass = 0
    currentImageCount = 0

    for dirpath, dirnames, filenames in os.walk(directory):

        for file in filenames:
            image = cv2.imread(os.path.join(dirpath, file))
            image = cv2.resize(image, (imageLength, imageWidth),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            labels.append(currentClass)
            currentImageCount = currentImageCount + 1
 
        print("{} Images in Class {}".format(currentImageCount, currentClass))
        currentImageCount = 0
        currentClass += 1

    images, labels = shuffle(images, labels)
    images = np.array(images)
    labels = np.array(labels)

    print('Done reading dataset with {} classes and {} images'.format(currentClass - 1, len(images)))

    print ("Saving Images and Labels to files")

    imagesFileName = dataset + "_Images" + "(" + str (imageLength) + ")" + ".npy"
    labelsFileName = dataset + "_Labels" + "(" + str (imageLength) + ")" + ".npy"

    np.save(imagesFileName, images)
    np.save(labelsFileName, labels)

    return images, labels