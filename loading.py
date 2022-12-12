import pickle
from sklearn.datasets import load_iris

# Step 4 - installed sklearn 1.1

# Step 5 - loaded pickled model
X_test, y_test = load_iris(return_X_y=True)
pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)

# Result should be 
# UserWarning: Trying to unpickle estimator LogisticRegression from version 1.0.2 when using version 1.1.3.
# This might lead to breaking code or invalid results. Use at your own risk.