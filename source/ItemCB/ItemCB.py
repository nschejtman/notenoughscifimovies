import re
import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

class ItemCB(BaseEstimator):

    def __init__(self, items_df):
        self.k = 5
        self.sh = 2
##        self.sim_mat =
        self.build_sim_mat(items_df)


    #Train recommender
    def fit(self, URM):
        print "fit"


    #Make predictions on trained recommender
    #Returns preidctions matrix
    def predict(self, users, n):
        #TODO: Consider only active ones
        print "pred"

    def build_sim_mat(self, items_df):
        aux_df = items_df.drop(['latitude', 'longitude', 'created_at'], axis=1)
        rows = aux_df.id.values
        aux_df = d.get_dummies(aux_df, columns=['career_level','discipline_id','industry_id','country', 'region','employment'])
        self.get_indices(aux_df, 'career_level')
        tf_idf_title = self.title_tfidf(aux_df)
        trans = TfidfTransformer()
        print self.get_indices(aux_df, 'career_level')
        tf_idf_career = trans.fit_transform(aux_df.career_level.dropna().values)
##        tf_idf_career = trans.fit_transform(aux_df.career_level.dropna().values)
##        tf_idf_career = trans.fit_transform(aux_df.career_level.dropna().values)
        
        print aux_df.columns, aux_df.head(1)

    def title_tfidf(self, aux_df):
        vectorizer = CountVectorizer()
        trans = TfidfTransformer()
        tf = vectorizer.fit_transform(aux_df.title.dropna().map(str).values)
        return trans.fit_transform(tf)

    def get_indices(self, aux_df, col_name):
        start, end = 0, 0
        cols = aux_df.columns.values
        for j in range(len(cols)):
            if col_name.match(cols[j]):
                start = j
                break
        for k in range(i, len(cols)):
            if col_name.match(cols[j]):
                end = k

        return (start, end)
        


#TODO: missing values
x = ItemCB(pd.read_csv('../../inputs/item_profile.csv', sep='\t'))
