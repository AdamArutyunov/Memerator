import cv2
from flask import Flask, abort, send_file
from flask import request as rq
from VGG import *
from GAN import *
from Constants import *

app = Flask(__name__)


@app.route("/recognize", methods=["POST"])
def recognize():
    files = rq.files

    image_file = files.get('image', None)
    if not image_file:
        return "Bad request: no input image.", 400

    image_file.save('temp/image')
    image = cv2.imread('temp/image')

    image = cv2.resize(image, (128, 128))
    image = image.astype("float") / 255.0

    #image = image.flatten()
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    preds = vgg.predict(image)

    i_max = preds.argmax(axis=1)[0]
    label = vgg_lb.classes_[i_max]

    return {
        'meme': label,
        'probability': float(preds[0][i_max])
    }


@app.route('/generate', methods=["GET", "POST"])
def generate():
    image = MyGAN.generate_image()
    print(image.shape)

    cv2.imwrite('temp/result.jpg', image)

    return send_file('temp/result.jpg')


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Creating VGGNet...")

    vgg = load_model('model/vgg/model')
    vgg_lb = pickle.loads(open('model/vgg/labels.pickle', "rb").read())

    print("VGGNet created.")
    print("Creating GAN...")

    MyGAN, MyDiscriminator, MyGenerator = create_gan()

    print('GAN created.')

    app.run('0.0.0.0', port=API_PORT)
