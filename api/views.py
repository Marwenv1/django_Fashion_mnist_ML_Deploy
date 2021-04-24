from django.shortcuts import render
import numpy as np
import re
import sys
## Apending MNIST model path
import os

from django_Fashion_mnist_ML_Deploy.settings import STATIC_URL

sys.path.append(os.path.abspath("./model"))
## custom utils file create for writing some helper func
from .utils import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

## Declaring global variable
global model, graph
## initializing MNIST model file (It comes from utils.py file)
# model, graph = init()
import base64
from PIL import Image
from io import BytesIO

## Declaring output path to save our image
OUTPUT = os.path.join(os.path.dirname(__file__), 'output.png')


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
    path = "D:/desktop/pullover.jpg"
    im = Image.open(os.path.join(path))
    convertImage(im)
    x = Image.open(OUTPUT, mode='L')
    x = np.invert(x)
    x = Image.resize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
    return JsonResponse({"output": response})


def index(request):
    return render(request, 'index.html', {})
