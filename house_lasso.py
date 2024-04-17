import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import pickle

df = pd.read_csv('house_data.csv')

columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

alpha=0.1
lasso = Lasso()
lasso.fit(X_train, y_train)

pickle.dump(lasso, open('model_lasso.pkl', 'wb'))
