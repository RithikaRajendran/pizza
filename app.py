import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv('pizza.csv')
x=df.drop(['buy'],axis=1)

y=df['buy']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [int(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features) 

    if prediction == 1:
        pred = "You have like pizza."
    elif prediction == 0:
        pred = "You don't like pizza."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
