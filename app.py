import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pickle

df = pd.read_csv("house_data.csv")
columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']

df = df[columns]

X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)


pickle.dump(lasso, open('model_lasso.pkl', 'wb'))
pickle.dump(ridge, open('model_ridge.pkl', 'wb'))


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
    

    return render_template('index.html', data_lasso=int(pred_lasso), data_ridge=int(pred_ridge), mse_ridge=mse_ridge , rmse_ridge=rmse_ridge, mae_ridge=mae_ridge, r2_ridge=r2_ridge, mse_lasso=mse_lasso, rmse_lasso=rmse_lasso, mae_lasso=mae_lasso, r2_lasso=r2_lasso )


if __name__ == '__main__':
    app.run(debug=True)