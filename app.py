from flask import Flask, render_template, request
import os,cv2
from keras.models import load_model
from keras.models import Model,load_model
from keras.applications.mobilenet import preprocess_input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D,Conv2D
from keras.layers.core import Activation,Flatten,Dense,Dropout
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_model():
    model=load_model('C:/Users/anand/Downloads/model360v6.h5')   
    print("model_loaded")
    return model

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload():
    model=get_model()
    target = os.path.join(APP_ROOT, 'static/xray/')
    if not os.path.isdir(target):
        os.mkdir(target)
    filename = ""
    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)
    img = cv2.imread(destination)
    
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(img)
    img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)
    img = img_to_array(img)
    img = cv2.resize(img,(360,360))
    img=img.reshape(1,360,360,3)
    img = img.astype('float32')
    img = img / 255.0
    pred = model.predict_classes(img)
    # prediction = int(np.argmax(pred, axis=1)[0])
    if pred:
        plot_dest = "/".join([target, "result.png"])

    return render_template("test.html", pred=pred, filename=filename)



if __name__ == '__main__':
    app.run(debug=True)
