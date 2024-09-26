from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)

scaler = pickle.load(open("Complete-Data-Science-With-Machine-Learning-And-NLP-2024\\5-Step By Step Project Implementation With LifeCycle Of ML Projects\\standard_scaler.pkl","rb"))
lr_model = pickle.load(open("Complete-Data-Science-With-Machine-Learning-And-NLP-2024\\5-Step By Step Project Implementation With LifeCycle Of ML Projects\\regression_model.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        scaled_new_data = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = lr_model.predict(scaled_new_data)

        return render_template("home.html",results=result[0])

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
