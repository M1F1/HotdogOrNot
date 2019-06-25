import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np
import flask 
import io
import os

app = flask.Flask(__name__)
model = None


def load_my_model(model_path):
    global model
    model = load_model(os.path.join(os.path.dirname(__file__), model_path))
    model._make_predict_function()          
    print('Model loaded. Start serving...')


def prepare_image(file_path, target_size):
    image = load_img(file_path, target_size=target_size)
    image = img_to_array(image)
    # mean and std from training
    image = (image -110.02) / 71.02
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
        return flask.render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        f = flask.request.files["file"]
        basepath = os.path.dirname(__file__)
        if not os.path.exists(os.path.join(basepath, 'uploads')):
            os.mkdir(os.path.join(basepath, 'uploads'))
        print('check ')
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image = prepare_image(file_path, target_size=(224, 224))

        preds = model.predict(image)
        numerical_pred = np.argmax(preds[0])
        result = "That is Hot-dog!"
        if numerical_pred == 0:
            result = "Not a Hot-dog!"
        return result
    return None 

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_my_model(model_path='models/keras_cnn_model.h5')
    app.run(host='0.0.0.0', port=80)
