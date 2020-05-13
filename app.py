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
    model=load_model('C:/Users/anand/Downloads/model128v5.h5')   
    print("model_loaded")
    return model
def Equalization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(img)
    return equalized_img
def Resize(img, size):
    img = img[:,1:-1]
    resized_image = cv2.resize(img, (size, size)) 
    return resized_image
def processimg(img):
    #     imgagePath='C:/Users/anand/Desktop/px27.jpg'
#     img = cv2.imread(imgagePath)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nbTrue=0
    n=0

    resized_image = Resize(img,360)
    equ = Equalization(resized_image)
    crop = CroppingLowerRegion(equ)

    height, width = crop.shape[:2]

    left, right = LungBoundary(crop) ### Binary Thresholding #Otsu Thresholding
    ret, thresholdImg = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresholdImg = cv2.bitwise_not(thresholdImg)

    LungAreaImg = np.zeros((height, width, 1), np.uint8)
    LungAreaImg = LungArea(LungAreaImg, left, right)
    crop=cv2.resize(crop,(360,360))
    return crop

@app.route("/")
def index():
    return render_template("index.html")

def predict(imgpath):
    img = cv2.imread(imgpath)
    img= processimg(img)
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img_to_array(img)
    img = cv2.resize(img,(360,360))
    img=img.reshape(1,360,360,3)
    img = img.astype('float32')
    img = img / 255.0
    result = model.predict_classes(img)
    pred=model.predict(img)
    return classes[result[0]]


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
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
    result = model.predict_classes(img)
    pred=model.predict(img)
    print(result)
    print(classes[result[0]])
    if pred:
        plot_dest = "/".join([target, "result.png"])
        

    return render_template("result.html", pred=pred,pos=pos,neg=neg, filename=filename)



if __name__ == '__main__':
    app.run(debug=True)
