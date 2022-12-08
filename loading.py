import pickle
from sklearn.datasets import load_iris

X_test, y_test = load_iris(return_X_y=True)
pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)