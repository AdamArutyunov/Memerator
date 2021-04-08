import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from imutils import paths
from miscellaneous import MyLabelBinarizer

matplotlib.use("Agg")
IMAGE_SIZE = (128, 128)


class SimpleNN:
    def __init__(self):
        self.data = []
        self.labels = []

        self.trainX = []
        self.testX = []
        self.trainY = []
        self.testY = []

        self.model = None

        self.last_epochs = -1
        self.train_history = []

        self.lb = MyLabelBinarizer()

    def create_model(self):
        model = Sequential()
        model.add(Dense(1024, input_shape=(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3,), activation="sigmoid"))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation="sigmoid"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.lb.classes_), activation="softmax"))

        self.model = model

        return model

    def load_model(self, model_path):
        model = load_model(model_path)
        self.model = model

        return model

    def fit(self, epochs, lr, batch_size):
        opt = SGD(lr=lr)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        self.last_epochs = epochs

        self.train_history = self.model.fit(self.trainX, self.trainY,
                                            validation_data=(self.testX, self.testY),
                                            epochs=epochs, batch_size=batch_size)

        predictions = self.model.predict(self.testX, batch_size=batch_size)
        print(classification_report(self.testY.argmax(axis=1), predictions.argmax(axis=1),
                                    target_names=self.lb.classes_))

    def cache_dataset(self, data_path, labels_path):
        with open(data_path, 'wb') as f:
            pickle.dump(self.data, f)

        with open(labels_path, 'wb') as f:
            pickle.dump(self.labels, f)

    def load_cached_dataset(self, data_path, labels_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)

        self.data = data
        self.labels = labels

        return data, labels

    def load_images(self, dataset_path, flatten=True, monochrome=False):
        data = []
        labels = []

        image_paths = list(paths.list_images(dataset_path))
        random.shuffle(image_paths)

        for image_path in image_paths:
            #print(image_path)
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE)

            if monochrome:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if flatten:
                image = image.flatten()

            data.append(image)

            label = image_path.split(os.path.sep)[-2]
            labels.append(label)

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        self.data = data
        self.labels = labels

        return data, labels

    def train_test_split(self, test_size=0.25):
        trainX, testX, trainY, testY = train_test_split(self.data, self.labels,
                                                        test_size=test_size, random_state=42)

        trainY = self.lb.fit_transform(trainY)
        testY = self.lb.fit_transform(testY)

        self.trainX = trainX
        self.testX = testX
        self.trainY = trainY
        self.testY = testY

        return trainX, testX, trainY, testY

    def save_output(self, output_model_path, output_labels_path, output_plot_path=None):
        self.model.save(output_model_path)
        f = open(output_labels_path, "wb")
        f.write(pickle.dumps(self.lb))
        f.close()

        if output_plot_path:
            N = np.arange(0, self.last_epochs)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.train_history.history["loss"], label="train_loss")
            plt.plot(N, self.train_history.history["val_loss"], label="val_loss")
            plt.plot(N, self.train_history.history["accuracy"], label="train_acc")
            plt.plot(N, self.train_history.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy (Simple NN)")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(output_plot_path)


if __name__ == "__main__":
    Model = SimpleNN()

    #Model.load_cached_dataset('datasets/cache/surfaces_data.pickle', 'datasets/cache/surfaces_labels.pickle')
    Model.load_images('datasets/memes')
    #Model.cache_dataset('datasets/cache/surfaces_data.pickle', 'datasets/cache/surfaces_labels.pickle')
    Model.train_test_split(0.25)
    print(len(Model.trainY))
    Model.create_model()
    Model.fit(300, 0.005, 16)

    Model.save_output('model/model.model', 'model/labels.pickle', 'model/plot.png')
