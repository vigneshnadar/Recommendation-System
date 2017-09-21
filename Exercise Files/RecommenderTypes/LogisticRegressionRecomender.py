# classification based cf system
# users who liked this also liked this based on user similarity attributes

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import sklearn as sk
from sklearn.linear_model import LogisticRegression

bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')

#print(bank_full.head())

#print(bank_full.info())

# you create dummy variables to represent the info in binary
# we create a model based on 19 attributes then we predict whether the new incoming user
# will buy the product or not
X=bank_full.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
# y_binary is the output
y=bank_full.ix[:,17].values
LogReg = LogisticRegression()
LogReg.fit(X,y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                   penalty='12', random_state=None, solver='liblinear', tol=0.0001,
                   verbose=0, warm_start=False)

new_user = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = LogReg.predict(new_user)
print(y_pred)

