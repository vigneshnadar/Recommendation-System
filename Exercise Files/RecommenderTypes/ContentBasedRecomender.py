#nearest neighbour algo. unsupervised classifier
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors

cars = pd.read_csv('mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt',
                'qsec', 'vs', 'am', 'gear', 'carb']
print(cars.head())

t = [15, 300, 160, 3.2]
X = cars.ix[:,(1, 3, 4, 6)].values

nbrs = NearestNeighbors(n_neighbors=1).fit(X)

print(nbrs.kneighbors([t]))
print(cars)


