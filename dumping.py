from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle

# Step 1 install sklearn 1.0

# Step 2 create LR model and fit the iris dataset
X, y = load_iris(return_X_y=True)
logistic_regression = LogisticRegression(random_state=0).fit(X, y)
print(logistic_regression.score(X, y))

# Step 3 - use pickle to dump the model 
pickle.dump(logistic_regression, open('model.pkl', 'wb'))






