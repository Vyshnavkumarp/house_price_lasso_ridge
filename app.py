from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model_lasso = pickle.load(open('model_lasso.pkl', 'rb'))
model_ridge = pickle.load(open('model_ridge.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['floors']
    val4 = request.form['yr_built']
    arr = np.array([val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred_lasso = model_lasso.predict([arr])
    pred_ridge = model_ridge.predict([arr])

    return render_template('index.html', data_lasso=int(pred_lasso), data_ridge=int(pred_ridge))

if __name__ == '__main__':
    app.run(debug=True)
