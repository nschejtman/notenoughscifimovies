import re
import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import logging
import heapq
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from numpy import linalg as LA

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def read_items():
    items_df = pd.read_csv('../../inputs/item_profile.csv', sep='\t')
    #items_df.career_level = items_df.career_level.fillna(0).astype('int64')
    return items_df



def cross_validate(items_df, URM, items_ids, n):
    params = {'k':[1, 2, 5, 10, 20, 50], 'sh':[1, 2, 3, 5, 10]}
    rec = GridSearchCV(ItemCB(items_df), params, scoring=map5_scorer, cv=3, fit_params={'item_ids':items_ids, 'n':n})
    reg.fit(URM, pos_items)
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
    def fit(self, URM, URM1, n):
        #self.URM_item_ids = list(item_ids)
        self.n = n
        #self.id_to_idx = {self.ICM_item_ids[i]:(i,item_ids.index(self.ICM_item_ids[i])) for i in range(URM.shape[1])}


    #Make predictions on trained recommender
    #Returns preidctions matrix
    def predict(self, URM):
        Y = [[] for _ in range(URM.shape[0])]
        for u in range(URM.shape[0]):
            rated = sps.nonzero(URM[u])
            rating = 0
            for i in range(URM.shape[1]):
                if self.actives[i] != 0 and URM[u,i] == 0:
                    closest = generate_knn(i, rated)
                    for j in closest:
                        rating += URM[u,j[1]]*j[0]
                        den += j[0]
                    rating /= den
                    if len(Y[u]) < self.n:
                        heapq.heappush(Y[u], (rating, i))
                    else:
                        if rating > Y[u][0]:
                            heapq.heappushpop(Y[u], (rating, i))

                    #TODO: SORT RECOMMENDATIONS

        recs = [[self.item_ids[i[1]] for i in row] for row in Y]

        return recs

    def generate_attrs(self, items_df):
        ''' Generates normalized vectors from the item-content matrix, using
        TF-IDF. Normalization is computed among the values corresponding to the
        same attribute in the original item-content matrix.

        Arguments:
        items_df -- item-content matrix
        '''

        self.item_ids = items_df.id.values
        self.actives = items_df.active_during_test.values
        self.attr_df = items_df.drop(['id', 'latitude', 'longitude', 'created_at', 'active_during_test'], axis=1)

        to_dummies = ['career_level','country', 'region','employment']
        to_tfidf = ['title', 'tags', 'discipline_id', 'industry_id']

        self.attr_df['career_level'] = self.attr_df['career_level'].fillna(0)
        self.attr_df['title'] = self.attr_df['title'].fillna('NULL').values
        self.attr_df['tags'] = self.attr_df['title'].fillna('NULL').values

        # Generate binary matrix
        self.attr_df = pd.get_dummies(self.attr_df, columns=to_dummies)

        self.attr_mat = {_:self.generate_tfidf(self.attr_df[_].map(str).values) for _ in to_tfidf}

        self.attr_df = self.attr_df.drop(to_tfidf, axis=1)

        logger.info(self.attr_df.columns)



    def generate_tfidf(self, data):
        vectorizer = CountVectorizer(token_pattern='\w+')
        trans = TfidfTransformer()
        tf = vectorizer.fit_transform(data)
        return trans.fit_transform(tf)#, vectorizer.vocabulary_)


    # Is this still necessary?
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

    def compute_similarity_matrix(self, norm_ICM) :
        ''' Computes the item-similarity matrix taking as input the normalized
        item-content matrix. Similarity computed as cosine similarity.

        Arguments:
        norm_ICM -- normalized item-content matrix data frame

        Returns:
        similarity matrix m of shape |I| x |I| where I is the set of items
        and m[i, j] is the cosine similarity between item i and j.'''

        # vi . vj
        numerator_matrix = norm_ICM.dot(norm_ICM.transpose())

        # Check for square matrix
        if (numerator_matrix.values.shape[0] != numerator_matrix.value.shape[1]):
            logger.error('The resulting similarity matrix is not square!')

        norm2_values = pd.DataFrame.from_dict(LA.norm(norm_ICM.values, 2, axis=0))

        # 1 / (|vi|2 |vj|2 + shrink)
        denominator_matrix = norm2_values.dot(norm2_values.transpose()).applymap(lambda x: 1/(x + self.sh))

        # Check for square matrix
        if (denominator_matrix.values.shape[0] != denominator_matrix.value.shape[1]):
            logger.error('The resulting similarity matrix is not square!')

        return numerator_matrix.dot(denominator_matrix.transpose())


    def sim(self, i, j):
        res = 0
        #num_atts = self.attr_df.shape[1] + reduce(lambda _, mat:_+mat.shape[1], self.attr_mat.itervalues(), 0)
        v_i, v_j = sps.lil_matrix(self.attr_df.iloc[[i]].values), sps.lil_matrix(self.attr_df.iloc[[j]].values)

        for _, v in self.attr_mat.items():
            v_i = sps.hstack([v_i, v[i]])
            v_j = sps.hstack([v_j, v[j]])

        return (v_i.dot(v_j.transpose())/(linalg.norm(v_i)*linalg.norm(v_j) + self.sh))[0,0]


    def calculate_knn(self, item, item_list):
        heap = []
        for it in item_list:
            aux = self.sim(item, it)
            if len(heap) < self.k:
                heapq.heappush(heap, (aux, it))
            else:
                if heap[0][0] < aux:
                    heapq.heappushpop(heap, (aux, it))
        return heap
        
##        self.neig_list = [[] for _ in range(self.attr_df.shape[0])]
##        for i in range(self.attr_df.shape[0]):
##            print i
##            self.neig_list[i] = []
##            for j in range(i, self.attr_df.shape[0]):
##                if j %1000 == 0:
##                    print j
##                aux = self.sim(i, j)
##                
##                        
##                if len(self.neig_list[j]) <= self.k:
##                    heapq.heappush(self.neig_list[j], (aux, i))
##                else:
##                    if self.neig_list[j][0][0] < aux:
##                        heapq.heappushpop(self.neig_list[j], (aux, i))









#TODO: missing values
items_df = read_items()
x = ItemCB(items_df)
