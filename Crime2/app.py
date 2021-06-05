# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:26:24 2020

@author: Administrator
"""
from flask import Flask,request,render_template
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=prediction[0]
    return render_template('index.html',prediction_text="Crime will be in {} UCR PART".format(output))
if __name__=="__main__":
    app.run(debug=True)
    