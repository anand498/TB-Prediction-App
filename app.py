from flask import Flask, render_template, request
import os,cv2
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
import numpy as np
import os
import sys
import numpy as np
import skimage.transform
import warnings
import cv2

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_model():
    model=load_model('C:/Users/anand/Downloads/model360v6.h5')   
    print("model_loaded")
    return model

@app.route("/")
def index():
    return render_template("index.html")

def remove_border(img, threshold=0):  #Crop image, throwing away the border below the threshold
    mask = img > threshold
    return img[np.ix_(mask.any(3), mask.any(0))]

def crop_center(img, size):   #Crop center sizexsize of the image
  
    y, x = img.shape #h,w,d
    startx = (x - size) // 15
    starty = (y - size) // 15
    return img[starty+200:starty-400, startx+200:startx+size-300]

def bigger_edge(img):
    y, x = img.shape
    return y if y < x else x

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
    cv2.imwrite('static/xray/file.png',img)

    
    img = cv2.resize(img,(360,360))
    img=img.reshape(1,360,360,3)
    img = img.astype('float32')
    img = img / 255.0
    pred = model.predict_classes(img)
    pred1=model.predict(img)
    pos=pred1[0][0]
    neg=pred1[0][1]
    if pred:
        plot_dest = "/".join([target, "result.png"])
        

    return render_template("result.html", pred=pred,pos=pos,neg=neg, filename=filename)



if __name__ == '__main__':
    app.run(debug=True)
