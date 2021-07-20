from flask import Flask, render_template, request
import cv2
import pickle
import numpy as np
from sklearn.decomposition import PCA

model = pickle.load(open('dogscats.pkl', 'rb'))
pca1 = pickle.load(open('pca.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def func():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    img_arr = []
    pixels = None
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)  

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixels = cv2.resize(img, (128,128)).flatten()
    img_arr.append(pixels)
    img_arr = np.array(img_arr)
    img_arr = img_arr/255.0

    img_arr_pca = pca1.transform(img_arr)
    pred = model.predict(img_arr_pca)

    return render_template('index.html', prediction=pred)



if __name__ == '__main__':
    app.run(port=3000, debug=True)