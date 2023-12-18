from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)

model = load_model('C:\Python\Final Project DSC UI\models\CNN-proker.h5')

def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    class_index = np.argmax(result)
    class_names = ['Differential', 'Evaporator', 'Spinner', 'Suspensi']
    print(result)
    return class_names[class_index]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', result='No file Uploaded')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', result='No Selected file')
        
        if file:
            file_path = "./static/" + file.filename
            file.save(file_path)
            result = predict_image(file_path)
            print(file.filename)
            return render_template('index.html', result=f'Predicted class: {result}', last_uploaded_photo = file.filename)
        

        
    except Exception as e:
        return render_template('index.html', result=f'Error:{e}')
 
if __name__ == '__main__':
    app.run(debug=True)