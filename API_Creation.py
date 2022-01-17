#!/usr/bin/env python
# coding: utf-8

# In[5]:

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request

model=tf.keras.models.load_model(r"C:\Users\theme\Downloads\PROJECTS\PERSONAL PROJECTS\Pnemonia Detection from Chest X-Rays\CHEST_XRAY.model")

def prepare_image(img):
    IMG_SIZE=80 # same as before
    img = Image.open(io.BytesIO(img))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    return img.reshape(-1,IMG_SIZE,IMG_SIZE,1)


def predict(img):
    Categories=["NORMAL","PNEUMONIA"]
    prediction=model.predict([prepare_image(img)])
    pred = np.argmax(prediction, axis=1)
    #return pred[0]
    return Categories[pred[0]]


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "file not found"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')


# In[ ]:




