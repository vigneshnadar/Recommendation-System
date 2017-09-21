import pandas as pd
import numpy as np

#ITEM BASED SIMILARITY. PEARSON R METHOD

frame = pd.read_csv('rating_final.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')
geodata = pd.read_csv('geoplaces2.csv', encoding = "ISO-8859-1")

places = geodata[['placeID', 'name']]
print(places.head())
rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())
print(rating.head())
rating = rating.sort_values('rating_count', ascending=False)
print(places[places['placeID']==135085])
print(cuisine[cuisine['placeID']==135085])

places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
print(places_crosstab.head())

Tortas_rating = places_crosstab[135085]
print(Tortas_rating[Tortas_rating>=0])

#evaluating similarity
similar_to_tortas = places_crosstab.corrwith(Tortas_rating)
corrTortas = pd.DataFrame(similar_to_tortas, columns=['PearsonR'])

corrTortas.dropna(inplace=True)
print(corrTortas.head())
Tortas_corr_summary = corrTortas.join(rating['rating_count'])
print(Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(10))


places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index=np.arange(7), columns=['placeID'])
summary = pd.merge(places_corr_Tortas, cuisine, on='placeID')
print(summary)