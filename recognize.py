# импортируем необходимые пакеты
from keras.models import load_model
from miscellaneous import MyLabelBinarizer
from imutils import paths
import argparse
import pickle
import cv2

 
model_path = 'model/model.model'
labels_path = 'model/labels.pickle'

width = 128
height = 128
flatten = 1

# загружаем модель и бинаризатор меток
print("[INFO] loading network and label binarizer...")
model = load_model(model_path)
lb = pickle.loads(open(labels_path, "rb").read())
 
imagePaths = sorted(list(paths.list_images("input")))

for imagePath in imagePaths:
    print(f"Image {imagePath}")
    image = cv2.imread(imagePath)
    output = image.copy()

    image = cv2.resize(image, (width, height))
    
    image = image.astype("float") / 255.0

    if flatten > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))
    else:
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    preds = model.predict(image)

    i_max = preds.argmax(axis=1)[0]
    print(preds)
    print(lb.classes_)
    label = lb.classes_[i_max]
    print(label)

    print("Result:")
    print(label, preds[0][i_max])
    print()

    text = "{}: {:.2f}%".format(label, preds[0][i_max] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 0, 0), 2)
    
    cv2.imwrite(f"output/{imagePath.lstrip('input/')}", output)
