import matplotlib.pyplot as plt
from SimpleNN import *
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from tqdm import tqdm
from miscellaneous import *

RANDOM_DIM = 100


class Generator(SimpleNN):
    def create_model(self):
        generator = Sequential()
        generator.add(Dense(256, input_dim=RANDOM_DIM, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(IMAGE_SIZE[0] * IMAGE_SIZE[1], activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=get_optimizer())

        self.model = generator

        return generator


class Discriminator(SimpleNN):
    def create_model(self):
        discriminator = Sequential()
        discriminator.add(Dense(512, input_dim=IMAGE_SIZE[0] * IMAGE_SIZE[1],
                                kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        #discriminator.add(Dense(512))
        #discriminator.add(LeakyReLU(0.2))
        #discriminator.add(Dropout(0.3))

        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=get_optimizer())

        self.model = discriminator

        return discriminator


class GAN:
    def __init__(self):
        self.model = None
        self.discriminator = None
        self.generator = None

    def load_models(self, discriminator, generator):
        self.discriminator = discriminator
        self.generator = generator

        discriminator.model.trainable = False

        gan_input = Input(shape=(RANDOM_DIM,))
        x = generator.model(gan_input)
        gan_output = discriminator.model(x)

        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=get_optimizer())

        self.model = gan

        return gan

    def generate_image(self):
        noise = np.random.normal(0, 1, size=[1, RANDOM_DIM])
        generated_image = self.generator.model.predict(noise)
        generated_image = generated_image.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1]) * 255

        return generated_image

    def plot_generated_images(self, epoch, generator, examples=9, dim=(3, 3), figsize=(10, 10)):
        noise = np.random.normal(0, 1, size=[examples, RANDOM_DIM])
        generated_images = generator.predict(noise) * (-1)
        generated_images = generated_images.reshape(examples, IMAGE_SIZE[0], IMAGE_SIZE[1])

        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'output/gan_generated_image_epoch_{epoch}.png')

    def train(self, epochs=1, batch_size=10):
        MySimpleNN = SimpleNN()
        try:
            raise Exception()
            MySimpleNN.load_cached_dataset("datasets/cache/memes_data.pickle", "datasets/cache/memes_labels.pickle")
        except Exception as e:
            MySimpleNN.load_images('datasets/memes_b', monochrome=True)
            MySimpleNN.cache_dataset("datasets/cache/memes_data.pickle", "datasets/cache/memes_labels.pickle")

        x_train, y_train, x_test, y_test = MySimpleNN.train_test_split()

        x_train = (x_train.astype(np.float32) - 0.5) * 2
        batch_count = x_train.shape[0] // batch_size

        for e in range(1, epochs + 1):
            print('-' * 15, 'Epoch %d' % e, '-' * 15)
            for _ in tqdm(range(batch_count)):
                noise = np.random.normal(0, 1, size=[batch_size, RANDOM_DIM])
                image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

                generated_images = self.generator.model.predict(noise)

                X = np.concatenate([image_batch, generated_images])

                y_dis = np.zeros(2 * batch_size)
                y_dis[:batch_size] = 0.9

                self.discriminator.trainable = True
                self.discriminator.model.train_on_batch(X, y_dis)

                noise = np.random.normal(0, 1, size=[batch_size, RANDOM_DIM])
                y_gen = np.ones(batch_size)

                self.discriminator.trainable = False
                self.model.train_on_batch(noise, y_gen)

            if e == 1 or e % 5 == 0:
                self.plot_generated_images(e, self.generator.model)


def create_gan():
    MyGAN = GAN()

    MyDiscriminator = Discriminator()
    MyDiscriminator.create_model()

    try:
        MyDiscriminator.model.load_weights('model/gan/discriminator/weights.h5')
    except Exception as e:
        print("Warning: unable to load Discriminator weights.")

    MyGenerator = Generator()
    MyGenerator.create_model()

    try:
        MyGenerator.model.load_weights('model/gan/generator/weights.h5')
    except Exception as e:
        print("Warning: unable to load Generator weights.")

    MyGAN.load_models(MyDiscriminator, MyGenerator)

    return MyGAN, MyDiscriminator, MyGenerator


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    MyGAN, MyDiscriminator, MyGenerator = create_gan()

    MyGAN.train(100, 16)

    MyDiscriminator.model.save_weights('model/gan/discriminator/weights.h5')
    MyGenerator.model.save_weights('model/gan/generator/weights.h5')

    print("Weights have saved.")
