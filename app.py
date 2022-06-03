import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
model = pickle.load(open('titanic_classifier.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plant')
def plant():
    return render_template('plant.html')


@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/typet')
def typet():
    return render_template('typet.html')

@app.route('/tools')
def tools():
    return render_template('tools.html')

@app.route('/process')
def process():
    return render_template('process.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    

    return render_template('result.html', prediction_text='The Passenger {}'.format(prediction[0]))

    


if __name__ == "__main__":
    app.run(debug=True)
    
    