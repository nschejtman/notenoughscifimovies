import random
import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import logging
import heapq
import time
import math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from numpy import linalg as LA
from collections import namedtuple

def map_scorer(rec, URM_test, hidden_ratings, n):
    score = 0
    rec_list = rec.predict(URM_test, n)
    i = 0
    for u in range(URM_test.shape[0]):
        if len(hidden_ratings[u]) > 0:
            is_relevant = np.in1d(rec_list[u], hidden_ratings[u], assume_unique=True)
            p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
            # assert len(rec_list[u]) > 0
            score += np.sum(p_at_k) / np.min([len(hidden_ratings[u]), len(rec_list[u])])
            i += 1
    return score / i if i != 0 else 0 #Really??

# ICM in rec constructor?
def CVSearch(rec, URM, ICM, actives, sample_size=None):
    np.random.seed(1)
    URM_sample = URM[np.random.permutation(URM.shape[0])[:sample_size],]
    params = {'k':[1,2,5,10],'sh':[0.5,2,5]}
    grid = list(ParameterGrid(params))
    folds = 4
    kfold = KFold(n_splits=folds, shuffle=True)  # Suffle UCM if necessary too, to keep indices correspondence
    splits = kfold.split(URM_sample)
    Result = namedtuple('Result', ['mean_score', 'std_dev', 'parameters'])
    results = []
    pred_ratings_perc = 0.75
    n = 5
    total = reduce(lambda acc, x: acc * len(x), params.itervalues(), 1) * folds
    prog = 1

    print "Progress: 0%"

    for pars in grid:
        rec = rec.set_params(**pars)
        maps = []
        for row_train, row_test in splits:
            URM_train = URM_sample[row_train,]
            rec.fit(URM_train, ICM, actives)
            URM_test = URM_sample[row_test,]
            hidden_ratings = []
            for u in range(URM_test.shape[0]):
                relevant_u = URM_test[u,].nonzero()[1]  # Indices of rated items for test user u
                # According to the distribution of interacton per user, excluding users with less than 2 jobs is the best choice
                if len(relevant_u) > 2:
                    # Randomly select some ratings to predict and hide other for the map score
                    np.random.shuffle(relevant_u)
                    URM_test[u,relevant_u[int(len(relevant_u) * pred_ratings_perc):]] = 0
                    hidden_ratings.append(relevant_u[int(len(relevant_u) * pred_ratings_perc):])
                else:
                    hidden_ratings.append([])
            maps.append(map_scorer(rec, URM_test, hidden_ratings, n))  # Assume rec to predict indices of items, NOT ids
            print "Progress: ",(prog * 100) / total, "%"
        results.append(Result(np.mean(maps), np.std(maps), pars))
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP5", "Std"] + _.parameters.keys())
    cols, col_feat, x_feat = 3, 'sh', 'k'
    f = sns.FacetGrid(data=scores, col=col_feat, col_wrap=cols, sharex=False, sharey=False)
    f.map(plt.plot, x_feat, 'MAP5')
    f.fig.suptitle("ItemCB CV MAP5 values")
    i_max, y_max = scores.MAP5.argmax(), scores.MAP5.max()
    i_feat_max = params[col_feat].index(scores[col_feat][i_max])
    f_max = f.axes[i_feat_max]
    f_max.plot(scores[x_feat][i_max], y_max, 'o', color='r')
    plt.figtext(0, 0, "COMMENT\nmaximum at (sh={:.5f},k={:.5f}, {:.5f}+/-{:.5f})".format(scores[col_feat][i_max],
                                                                                         scores[x_feat][i_max],
                                                                                         y_max,
                                                                                         scores['Std'][i_max]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    f.savefig('ItemCB CV MAP5 values.png', bbox_inches='tight')
    return scores

def HoldoutSearch(rec, URM, ICM, actives, sample_size=None):
    np.random.seed(1)
    nnz = URM.nonzero()
    perm = np.random.permutation(len(nnz[0]))[:sample_size]
    unique_rows = np.unique(nnz[0][perm])
    nnz_to_sample_row = {unique_rows[i]:i for i in range(len(unique_rows))}
    URM_sample = sps.lil_matrix((unique_rows.size, URM.shape[1]))
    for i in perm:
        URM_sample[nnz_to_sample_row[nnz[0][i]], nnz[1][i]] = URM[nnz[0][i], nnz[1][i]]
    train_size = 0.75
    params = {'k': [5], 'sh': [2]}
    grid = list(ParameterGrid(params))
    repetitions = 4
    Result = namedtuple('Result', ['mean_score', 'std_dev', 'parameters'])
    results = []
    n = 5
    total = reduce(lambda acc, x: acc * len(x), params.itervalues(), 1) * repetitions
    prog = 1

    print "Progress: 0%"

    for pars in grid:
        rec = rec.set_params(**pars)
        maps = []
        for rep in range(repetitions):
            np.random.shuffle(perm)
            URM_train, URM_test = URM_sample.copy(), URM_sample.copy()
            pivot = int(len(perm) * train_size)
            for i in range(len(perm)):
                if i < pivot:
                    URM_test[nnz_to_sample_row[nnz[0][perm[i]]], nnz[1][perm[i]]] = 0
                else:
                    URM_train[nnz_to_sample_row[nnz[0][perm[i]]], nnz[1][perm[i]]] = 0
            rec.fit(URM_train, ICM, actives)
            hidden_ratings, test_rows = [], []
            for u in range(URM_test.shape[0]):
                relevant_u = URM_test[u,].nonzero()[1]  # Indices of rated items for test user u
                if len(relevant_u) > 0:
                    hidden_ratings.append(relevant_u)
                    test_rows.append(u)
            maps.append(map_scorer(rec, URM_train[test_rows,], hidden_ratings, n))  # Assume rec to predict indices of items, NOT ids
            print "Progress: ", (prog * 100) / total, "%"
            prog += 1
        results.append(Result(np.mean(maps), np.std(maps), pars))
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP5", "Std"] + _.parameters.keys())
    cols, col_feat, x_feat = 3, 'sh', 'k'
    f = sns.FacetGrid(data=scores, col=col_feat, col_wrap=cols, sharex=False, sharey=False)
    f.map(plt.plot, x_feat, 'MAP5')
    f.fig.suptitle("ItemCB Holdout MAP5 values")
    i_max, y_max = scores.MAP5.argmax(), scores.MAP5.max()
    i_feat_max = params[col_feat].index(scores[col_feat][i_max])
    f_max = f.axes[i_feat_max]
    f_max.plot(scores[x_feat][i_max], y_max, 'o', color='r')
    plt.figtext(0, 0, "COMMENT\nmaximum at (sh={:.5f},k={:.5f}, {:.5f}+/-{:.5f})".format(scores[col_feat][i_max],
                                                                                         scores[x_feat][i_max],
                                                                                         y_max,
                                                                                         scores['Std'][i_max]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    f.savefig('ItemCB Holdout MAP5 values.png', bbox_inches='tight')
    return scores

def read_items():
    items_df = pd.read_csv('../../inputs/item_profile.csv', sep='\t')
    # items_df.career_level = items_df.career_level.fillna(0).astype('int64')
    return items_df

def read_interactions():
    ints = pd.read_csv('../../inputs/interactions_idx.csv', sep='\t')
    return sps.csr_matrix((ints['interaction_type'].values, (ints['user_idx'].values, ints['item_idx'].values)))

def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)

def read_top_pops():
    aux = pd.read_csv('../../inputs/top_pop_ids.csv', sep='\t')['0'].values
    return aux

class ItemCB(BaseEstimator):
    def __init__(self, k=5, sh=2, k_top=500): #k_rated??
        self.k = k
        self.sh = sh
        self.k_top = k_top
        self.pop = None
        self.ICM = None
        self.sim_mat = None


    # Train recommender
    def fit(self, URM, ICM, actives):
        """
        Arguments:
        n -- number of recommendations
        pos_items -- list of item indices rated by users (used by GridSeahCV)
        URM -- user rating matrix
        """
        self.ICM = sps.csr_matrix(ICM) # This makes a reference to ICM, not a copy
        self.sim_mat = self.compute_similarity_matrix(ICM, partitions_number=17)

    # Make predictions on trained recommender
    # Returns preidctions matrix with indices
    def predict(self, URM, n):
        # recs = [[] for _ in range(URM.shape[0])] #TODO: Optimize inside for
        # for u in range(URM.shape[0]):
        #     st = time.time()
        #     rated = URM[u].nonzero()[1]
        #     if len(rated) == 0:
        #         recs[u] = list(self.pop[:n])  # [self.item_ids[i] for i in self.pop[:self.n]]
        #     else:
        #         for i in self.pop:
        #             if URM[u, i] == 0:
        #                 closest = self.calculate_knn(i, rated)
        #                 den = 0
        #                 rating = 0
        #                 for j in closest:
        #                     rating += URM[u, j[1]] * j[0]
        #                     den += j[0]
        #                 rating /= 1 if den == 0 else den
        #
        #                 if len(recs[u]) < n:
        #                     heapq.heappush(recs[u], (rating, i))
        #                 else:
        #                     if rating > recs[u][0][0]:
        #                         heapq.heappushpop(recs[u], (rating, i))
        #         recs[u].sort(reverse=True) #We were puttinh the in incresing order!!
        #         recs[u] = [i[1] for i in recs[u]]  # [self.item_ids[i[1]] for i in Y[u]]
        #     assert len(recs[u]) > 0
        #     et = time.time()
        #
        #
        # return recs
        user_profile = URM
        scores = user_profile.dot(self.sim_mat.T).ravel()

        if True:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(user_profile.data)
            den = rated.dot(self.sim_mat.T).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den

        nonzero_indices = user_profile.nonzero()
        scores[nonzero_indices[0], nonzero_indices[1]] = 0.0

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:n]


    def sim(self, i, j):
        v_i, v_j = self.ICM[i,], self.ICM[j,]
        return (v_i.dot(v_j.transpose()) / (linalg.norm(v_i) * linalg.norm(v_j) + self.sh))[0, 0]

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

    def compute_similarity_matrix(self, matrix, row_wise=True, partitions_number=5):
        matrix = check_matrix(matrix, format='csr' if row_wise else 'csc')
        matrix_norms = matrix.copy()
        matrix_norms.data **= 2

        matrix_norms = matrix_norms.sum(axis=1 if row_wise else 0)

        matrix_norms = np.asarray(np.sqrt(matrix_norms)).ravel()
        matrix_norms += 1e-6
        repetitions = np.diff(matrix.indptr)

        matrix_norms = np.repeat(matrix_norms, repetitions)
        matrix.data /= matrix_norms
        sim = None
        if row_wise:
            psize = matrix.shape[0] / partitions_number
            for i in range(partitions_number + (matrix.shape[0] % partitions_number)):
                start = i * psize
                end = start + psize - 1 if i != partitions_number + (matrix.shape[0] % partitions_number) else matrix.shape[0]
                partitioned_matrix = matrix[start:end, ]
                similarity_matrix = partitioned_matrix.dot(matrix.T)
                if self.sh > 0:
                    similarity_matrix = self.apply_shrinkage(partitioned_matrix, similarity_matrix)

                idx_sorted = np.argsort(similarity_matrix, axis=1)
                not_top_k = idx_sorted[:, :-self.k]
                similarity_matrix[np.arange(similarity_matrix.shape[0]), not_top_k] = 0.0
                if i == 0 :
                    sim = similarity_matrix.copy()
                else :
                    sim = sps.vstack([sim, similarity_matrix])
        else :
            #TODO
            pass
        return sim




    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind.T.dot(X_ind).toarray().astype(np.float32)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        dist *= co_counts / (co_counts + self.sh)
        return dist







def generate_attrs(items_df):
    ''' Generates normalized vectors from the item-content matrix, using
    TF-IDF. Normalization is computed among the values corresponding to the
    same attribute in the original item-content matrix.

    Arguments:
    items_df -- item-content matrix
    '''
    attr_df = items_df.drop(['id', 'latitude', 'longitude', 'created_at', 'active_during_test', 'title', 'tags'],
                            axis=1)
    attr_mats = {}
    to_dummies = ['career_level', 'country', 'region', 'employment']
    to_tfidf = ['discipline_id', 'industry_id']

    attr_df['career_level'] = attr_df['career_level'].fillna(0)
    #   Concatenate title and tags? Are correlated
    #   attr_df['title'] = attr_df['title'].fillna('NULL').values
    #   attr_df['tags'] = attr_df['tags'].fillna('NULL').values

    # Generate binary matrix
    trans = CountVectorizer(token_pattern='\w+')
    for attr in to_dummies:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()

    # attr_mats = {_: generate_tfidf(attr_df[_].map(str).values) for _ in to_tfidf}
    trans = TfidfVectorizer(token_pattern='\w+')
    for attr in to_tfidf:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()


    return reduce(lambda acc, x: x if acc.shape == (1, 1) else sps.hstack([acc, x]), attr_mats.itervalues(),
                  sps.lil_matrix((1, 1)))

items_df = read_items()
URM = read_interactions()
# test_users = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values[:10]
# urm = urm[test_users,]
actives = items_df.active_during_test.values
item_ids = items_df.id.values
ICM = generate_attrs(items_df)
rec = ItemCB(k_top=62)
# CVSearch(rec, URM, ICM, actives, 4000)
# HoldoutSearch(rec, URM, ICM, actives, 40000)
# a.fit(urm, None, 5, items_df)
rec.fit(URM, ICM, actives)
recs = rec.predict(URM[:1000], 5)
print recs
# user_df = pd.read_csv('../../inputs/user_profile.csv', sep='\t')
# out_file = open('../../output/ItemCB Nico.csv', 'wb')
# out_file.write('user_id,recommended_items\n')
# for i in range(len(recs)):
# ##    aux = ''
# ##    for _ in recs[i]:
# ##        aux += _ + ' '
#     out_file.write(str(user_df.loc[test_users[i]]['user_id']) + ',' + reduce(lambda acc, x: acc+str(x) + ' ', recs[i], '') + '\n')
# out_file.close()
# ##pd.DataFrame(recs, index=[user_df.loc[i]['user_id'] for i in test_users[:len(test_users)/2]]).to_csv('../../output/ItemCB2.csv', sep=' ', index=True, header=False)
# st = time.time()
# #cross_validate(items_df, urm, 5)
# et = time.time()
