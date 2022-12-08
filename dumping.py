from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle

# step 1 install sklearn 1.0

# step 2 and 3 create LR model and fit the iris dataset
X, y = load_iris(return_X_y=True)
logistic_regression = LogisticRegression(random_state=0).fit(X, y)
print(logistic_regression.score(X, y))

# step 4 use pickle to dump the model 
pickle.dump(logistic_regression, open('model.pkl', 'wb'))




