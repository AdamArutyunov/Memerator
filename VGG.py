from SimpleNN import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


class VGG(SimpleNN):
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
        model.add(MaxPooling2D((2, 2), strides=2))
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(self.lb.classes_), activation="softmax"))

        self.model = model

        return model

    def load_images(self, dataset_path, flatten=True, monochrome=False):
        super().load_images(dataset_path, False, monochrome)


if __name__ == "__main__":
    Model = VGG()

    #Model.load_cached_dataset('datasets/cache/surfaces_data.pickle', 'datasets/cache/surfaces_labels.pickle')
    Model.load_images('datasets/memes')
    #Model.cache_dataset('datasets/cache/surfaces_data.pickle', 'datasets/cache/surfaces_labels.pickle')
    Model.train_test_split(0.25)
    print(len(Model.trainY))
    Model.create_model()
    Model.fit(10, 0.01, 1)

    Model.save_output('model/vgg/model', 'model/vgg/labels.pickle')