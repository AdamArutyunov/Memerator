import os
from PIL import Image, ImageFilter
from imutils import paths

input_images_path = "datasets/memes_b"


def rename():
    for i, image_path in enumerate(list(paths.list_images(input_images_path))):
        image_path = image_path.replace('\\', '/')
        path = '/'.join(image_path.split('/')[:-1])
        format = image_path.split('.')[-1]
        new_name = f'{path}/{i}.{format}'
        print(new_name)
        os.rename(image_path, f'{path}/{i}.{format}')


def multiply():
    for imagePath in list(paths.list_images(input_images_path)):
        try:
            image = Image.open(imagePath)
            image = image.resize((128, 128))
            imagePath = imagePath.split(".")
            image.save(imagePath[0] + "_resized." + imagePath[-1])
        except Exception as e:
            pass

    for imagePath in list(paths.list_images(input_images_path)):
        try:
            image = Image.open(imagePath)
            image = image.filter(ImageFilter.BLUR)
            image = image.filter(ImageFilter.BLUR)
            imagePath = imagePath.split(".")
            image.save(imagePath[0] + "_blur." + imagePath[-1])
        except Exception as e:
            pass


multiply()