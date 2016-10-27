import re
import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


def read_items():
    items_df = pd.read_csv('../../inputs/item_profile.csv', sep='\t')
    #items_df.career_level = items_df.career_level.fillna(0).astype('int64')
    return items_df



def cross_validate(items_df, URM):
    params = {'k':[1, 2, 5, 10, 20, 50], 'sh':[1, 2, 3, 5, 10]}
    rec = GridSearchCV(ItemCB(items_df), params, scoring=map5_scorer, cv=5)
    reg.fit(X, Y)
    scores = pd.DataFrame(data=[[_.mean_validation_score,np.std(_.cv_validation_scores)]+ _.parameters.values() for _ in reg.grid_scores_],
                          columns=["MAP5", "Std"] + _.parameters.keys())
    cols, col_feat, x_feat = 3, 'sh', 'k'
    f = sns.FacetGrid(data=scores, col=col_feat, col_wrap=cols, sharex=False, sharey=False)
    f.map(plt.plot, x_feat, 'MAP5')
    f.fig.suptitle("ItemCB CV MAP5 values")
    i_min, y_min = scores.MAP5.argmin(), scores.MAP5.min()
    i_feat_min = params[col_feat].index(scores[col_feat][i_min])
    f_min = f.axes[i_feat_min]
    f_min.plot(scores[x_feat][i_min], y_min, 'o', color='r')
    plt.figtext(0, 0, "COMMENT\nMinimum at (sh={:.5f},k={:.5f}, {:.5f}+/-{:.5f})".format(scores[col_feat][i_min],                                                                 
                                                                                         scores[x_feat][i_min],
                                                                                         y_min,
                                                                                         scores['Std'][i_min]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    f.savefig('ItemCB CV MAP5 values.png', bbox_inches='tight')
    return scores


    

class ItemCB(BaseEstimator):

    def __init__(self, items_df):
        self.k = 5
        self.sh = 2
        self.generate_attrs(items_df)


    #Train recommender
    def fit(self, URM):
        print "fit"


    #Make predictions on trained recommender
    #Returns preidctions matrix
    def predict(self, users, n):
        #TODO: Consider only active ones
        print "pred"



    def generate_attrs(self, items_df):
        self.item_ids = items_df.id.values
        self.attr_df = items_df.drop(['id', 'latitude', 'longitude', 'created_at'], axis=1)

        to_dummies = ['career_level','country', 'region','employment']
        to_tfidf = ['title', 'tags', 'discipline_id', 'industry_id']

        self.attr_df = pd.get_dummies(self.attr_df, columns=to_dummies)
        self.attr_mat = {}
        #TODO: aggregate in one line
        self.attr_mat['title'] = self.generate_tfidf(self.attr_df['title'].dropna().values) #NAs!!
        self.attr_mat['tags'] = self.generate_tfidf(self.attr_df['tags'].dropna().values)
        self.attr_mat['discipline_id'] = self.generate_tfidf(self.attr_df['discipline_id'].dropna().astype(int).map(str).values)
        self.attr_mat['industry_id'] = self.generate_tfidf(self.attr_df['industry_id'].dropna().astype(int).map(str).values)

        self.attr_df = self.attr_df.drop(to_tfidf, axis=1)
        
        print self.attr_df.columns



    def generate_tfidf(self, data):
        vectorizer = CountVectorizer(token_pattern='\w+')
        trans = TfidfTransformer()
        tf = vectorizer.fit_transform(data)
        return (trans.fit_transform(tf), vectorizer.vocabulary_)
    


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
items_df = read_items()
x = ItemCB(items_df)
