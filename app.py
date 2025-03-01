from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model

app = Flask(__name__)

# Function to handle prediction based on the model
def predict(values, dic):
    if len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)

@app.route("/recommendationPage")
def recommendationPage():
    kidney_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Sample values for kidney prediction
    pred_kidney = predict(kidney_values, {})  # Use actual logic

    return render_template('recommendations.html', pred_kidney=pred_kidney)

@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia_predict.html', pred=pred)

@app.route('/healthy_tips')
def healthy_tips():
    return render_template('healthy_tips.html')

@app.route('/stages', methods=['GET', 'POST'])
def stages():
    # Render the page without any result by default
    return render_template('stages.html', stage=None, error_message=None)

@app.route("/calculate_stage", methods=['POST'])
def calculate_stage():
    egfr = request.form.get('egfr', '')
    stage = None
    error_message = None

    # Validate eGFR input and calculate the stage
    try:
        egfr = float(egfr)
        if egfr >= 90:
            stage = "Stage 1 - Normal Kidney Function"
        elif 60 <= egfr < 90:
            stage = "Stage 2 - Mild Decrease in Kidney Function"
        elif 30 <= egfr < 60:
            stage = "Stage 3 - Moderate Decrease in Kidney Function"
        elif 15 <= egfr < 30:
            stage = "Stage 4 - Severe Decrease in Kidney Function"
        else:
            stage = "Stage 5 - Kidney Failure"
    except ValueError:
        error_message = "Please enter a valid numeric eGFR value."

    # Render stages.html with the result or error message
    return render_template('stages.html', stage=stage, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
