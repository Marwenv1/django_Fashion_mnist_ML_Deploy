import time

from keras.preprocessing import image


from PIL import Image


from django.shortcuts import render
import numpy as np
import re
import sys
## Apending MNIST model path
import os
from numpy import asarray

from django_Fashion_mnist_ML_Deploy.settings import STATIC_URL

sys.path.append(os.path.abspath("./model"))
## custom utils file create for writing some helper func
from .utils import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

## Declaring global variable
global model, graph
## initializing MNIST model file (It comes from utils.py file)
model, graph = init()
import base64
from PIL import Image
from io import BytesIO

from keras.preprocessing.image import load_img
import warnings
## Declaring output path to save our image
OUTPUT = os.path.join(os.path.dirname(__file__), 'output.png')
JSONpath = os.path.join(os.path.dirname(__file__), 'models', 'model.json')
MODELpath = os.path.join(os.path.dirname(__file__), 'models', 'mnist.h5')

def getI420FromBase64(codec):
    base64_data = re.sub('^data:image/.+;base64,', '', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    img.save(OUTPUT)


def convertImage(imgData):
    getI420FromBase64(imgData)


@csrf_exempt
def predict(request):
    imgData = request.POST.get('image')
    # path = "C:/Users/marwe/OneDrive/Desktop/idk/man.jpg"
    convertImage(imgData)
    # load the image
    print(OUTPUT)
    image = Image.open(OUTPUT).convert('L')
    image = image.resize((28,28),Image.ANTIALIAS)
    # convert image to numpy array
    data = asarray(image)
    print(type(data))
    # summarize shape
    print(data.shape)
    data.reshape(-1,28, 28, 1)
    print(data.shape)


    with graph.as_default():
        json_file = open(JSONpath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(MODELpath)
        print("Loaded Model from disk")
        # compile and evaluate loaded model
        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        data = np.expand_dims(data, axis=0)
        print(data.shape)  # (1, 28, 28)
        data = np.expand_dims(data, axis=3)
        print(data.shape)  # (1, 28, 28,1)


        out = loaded_model.predict(data)

        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))

    return JsonResponse({"output": response})


def index(request):
    return render(request, 'index.html', {})
