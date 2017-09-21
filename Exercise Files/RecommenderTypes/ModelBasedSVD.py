#mATRIX Factorization from SVD
import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD


columns = ['user_id', 'item_id', 'rating', 'timestamp']
frame = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
#print(frame.head())

columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL',
           'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
           'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
           'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
#combne movie and their review
combined_movies_data = pd.merge(frame, movies, on='item_id')
#print(combined_movies_data.head())

combined_movies_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head()
filter = combined_movies_data['item_id']==50
print(combined_movies_data[filter]['movie title'].unique())


#building a utility matrix
rating_crosstab = combined_movies_data.pivot_table(values = 'rating', index = 'user_id', columns = 'movie title',
                                                   fill_value=0)
print(rating_crosstab.head())

#transposing the matrix
rating_crosstab.shape
#transposing the matrix
X = rating_crosstab.values.T
print(X.shape)


#decomposing the matrix
SVD =TruncatedSVD(n_components=12, random_state=17)

resultant_matrix = SVD.fit_transform(X)
resultant_matrix.shape

# making the reco
# so basically pearson r creates a movie to movie correlation matrix
# you select the movie which correlates the most with your movie of interest based on generalized user tastes

#generate a correlation matrix
corr_mat = np.corrcoef(resultant_matrix)
print(corr_mat.shape)

#isolating star wars from corr matrix
movies_names = rating_crosstab.columns
movies_list = list(movies_names)

star_wars = movies_list.index('Star Wars (1977)')
print(star_wars)
corr_star_wars = corr_mat[star_wars]
print(corr_star_wars.shape)
print(list(movies_names[(corr_star_wars < 1.0) & (corr_star_wars > 0.9)]))





