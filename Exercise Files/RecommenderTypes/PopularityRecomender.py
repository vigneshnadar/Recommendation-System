import pandas as pd
import  numpy as np

frame = pd.read_csv('rating_final.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')

print(frame.head())
print(cuisine.head())

rating_count = pd.DataFrame(frame.groupby('placeID')['rating'].count())

print(rating_count.sort_values('rating', ascending=False).head())
most_rated_places = pd.DataFrame([135085, 132825, 135032, 135052, 132834], index=np.arange(5), columns=['placeID'])
summary = pd.merge(most_rated_places, cuisine, on='placeID')

print(cuisine['Rcuisine'].describe())



