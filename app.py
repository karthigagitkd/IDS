# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:00:44 2023

@author: Karthiga Devi K
"""
# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'ids.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("index.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        id= int(request.form['id'])
        dur = float(request.form.get['dur'])
        proto = request.form.get('proto')
        # service = request.form.get('service')
        # spkts = int(request.form['spkts'])
        # dpkts = int(request.form['dpkts'])
        # sbytes = int(request.form['sbytes'])
        # dbytes = int(request.form['dbytes'])
        # rate = float(request.form['rate'])
        # sttl = int(request.form['sttl'])
        # dttl = int(request.form['dttl'])
        # sload = float(request.form['sload'])
        # dload = float(request.form['dload'])
        # sloss = int(request.form['sloss'])
        # dloss = int(request.form['dloss'])
        # fb = request.form.get('fbs')
        # restecg = int(request.form['restecg'])
        # thalach = int(request.form['thalach'])
        # exang = request.form.get('exang')
        # oldpeak = float(request.form['oldpeak'])
        # slope = request.form.get('slope')
        # ca = int(request.form['ca'])
        # thal = request.form.get('thal')
        
        data = np.array([[id,dur,proto]])
        my_prediction = model.predict(data)   
        return render_template("index.html", prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run()


       