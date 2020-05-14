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

size=360

def Resize(img, size):
    img = img[:,1:-1]
    resized_image = cv2.resize(img, (size, size)) 
    return resized_image
def CroppingLowerRegion(img):
    height, width = img.shape[:2]
    x = height-1
    stop = True
    lastMean = 0
    #while we aren't in the horizontal middle of the image or the lung start isn't find
    while(x > (height/2) and stop):
        y = 20
        lastMean = 0
        #we move on the ligne
        while(y < width/2 and stop):
            #we take the mean of 5 pixel
            batche = img[x,y:y+5]
            mean = cv2.mean(batche)
            #if the gradient beetween the lastMean and the actual is greater than 50 we stop
            if(lastMean - mean[0] > 50):
                stop = False
                break
            else:
                lastMean = mean[0]
            y+= 5
        x -= 2;
    img = img[0:x,0:width-1]
    return img

def LungBoundary(img):
    height, width = img.shape[:2]
    #we cut the image into 2 image
    imgLeft = img[0:height,0:int(width/2)]
    imgRight = img[0:height,int(width/2):width-1]
    #we create a range of equidistant lines
    r = range(height-1,0, -int((height)/50))
    i = 0;
    #Left Variable
    lastYLeft=-50
    arrayLeft = []
    mostRealLeft =-50

    #Right Variable
    lastYRight=-50
    arrayRight = []
    mostRealRight =-50

    #for each lines
    for x in r:
        #leftPart
        y = 20
        lastMean = 0 
        heightLeft, widthLeft = imgLeft.shape[:2]
        while(y < widthLeft-6):
            batche = imgLeft[x,y:y+5] 
            mean = cv2.mean(batche)
            #if the gradient beetween the two batch is greater than 15 
            if(lastMean - mean[0] > 15):
                #if lastYLeft isn't initialize we initialize is value
                if(lastYLeft < 0):
                    lastYLeft = y
                    mostRealLeft = y
                    break
                else:
                    if(mostRealLeft == -50):
                        mostRealLeft = y
                    #we compare the new possible y with the last find and keep the most close of the y of the previous line
                    elif(abs(mostRealLeft - lastYLeft) > abs(y - lastYLeft)):
                        mostRealLeft = y;
            lastMean = mean[0]
            y+= 5
        #imgLeft[x-3:x+3,mostRealLeft-3:mostRealLeft+3] = 255
        arrayLeft.append((mostRealLeft,x))
        if(mostRealLeft != -50):
            lastYLeft = mostRealLeft
        mostRealLeft = -50;


        #RighPart same as for left part
        lastMean = 0 
        heightRight, widthRight = imgRight.shape[:2]
        y = widthRight-1;
        while(y > 6):
            batche = imgRight[x,y-5:y] 
            mean = cv2.mean(batche)
            if(lastMean - mean[0] > 15):
                if(lastYRight < 0):
                    lastYRight = y
                    mostRealRight = y
                    break
                else:
                    if(mostRealRight == -50):
                        mostRealRight = y-5
                    elif(abs(mostRealRight - lastYRight) > abs(y-5 - lastYRight)):
                        mostRealRight = y-5;
            lastMean = mean[0]
            y-= 5
        #imgRight[x-3:x+3,mostRealRight-3:mostRealRight+3] = 255
        arrayRight.append((mostRealRight + widthRight,x))
        if(mostRealRight != -50):
            lastYRight = mostRealRight
        
        mostRealRight = -50;

        i += 1
    return arrayLeft,arrayRight


#create a mash of the lung area with theleft and right part of the picture
def LungArea(LungAreaImg, left, right):
    height, width = LungAreaImg.shape[:2]
    lastCoord = (0, height-1)

    for x in left:
        cv2.rectangle(LungAreaImg, x, lastCoord, 255, -1)
        lastCoord = (0, x[1])
    cv2.rectangle(LungAreaImg, (0, 0), (x[0], x[1]), 255, -1)

    lastCoord = (width-1, height-1)
    for x in right:
        cv2.rectangle(LungAreaImg, x, lastCoord, 255, -1)
        lastCoord = (width-1, x[1])
    cv2.rectangle(LungAreaImg, (width-1, 0), (x[0], x[1]), 255, -1)
    
    LungAreaImg = cv2.bitwise_not(LungAreaImg)
    
    return LungAreaImg


#compare the lung area w=before and after the otsu threshold to determine if the lung is infected or not
def Compare(crop, LungAreaImg, thresholdImg):
    thresholdMask = cv2.bitwise_and(thresholdImg, LungAreaImg)
    cropThreshold = cv2.bitwise_and(crop, thresholdMask)
    
    nbPixelArea = cv2.countNonZero(LungAreaImg)
    nbPixelThreshold = cv2.countNonZero(thresholdMask)

    return (((nbPixelArea - nbPixelThreshold) / nbPixelArea)  < 0.62),cropThreshold

def processimg(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nbTrue=0
    n=0
    # resized_image = Resize(img,size)
    img = cv2.resize(img, (size, size)) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(img)
    crop = CroppingLowerRegion(equalized_img)
    height, width = crop.shape[:2]
    left, right = LungBoundary(crop) ### Binary Thresholding #Otsu Thresholding
    ret, thresholdImg = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresholdImg = cv2.bitwise_not(thresholdImg)
    LungAreaImg = np.zeros((height, width, 1), np.uint8)
    LungAreaImg = LungArea(LungAreaImg, left, right)
    crop=cv2.resize(crop,(size,size))
    return crop

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/upload", methods=['POST'])
def upload():
    model=load_model('C:/Users/anand/Downloads/model128v5.h5')   
    print("model_loaded")
    # model=get_model()
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
    img= processimg(img)
    cv2.imwrite('static/xray/processedfile.png',img)
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img_to_array(img)
    img = cv2.resize(img,(size,size))
    img=img.reshape(1,size,size,3)
    img = img.astype('float32')
    img = img / 255.0
    result = model.predict_classes(img)
    pred=model.predict(img)
    neg=pred[0][0]
    pos=pred[0][1]
    classes=['Negative','Positive']
    predicted=classes[result[0]]
    plot_dest = "/".join([target, "result.png"])

    return render_template("result.html", pred=predicted,pos=pos,neg=neg, filename=filename)



if __name__ == '__main__':
    app.run(debug=True)
