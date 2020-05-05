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