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
            # print u, rec_list[u], len(rec_list[u]), len(rec_list), URM.shape
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
                    #print relevant_u, hidden_ratings[u], URM_test[u,].nonzero()[1]
                else:
                    hidden_ratings.append([])
            maps.append(map_scorer(rec, URM_test, hidden_ratings, n))  # Assume rec to predict indices of items, NOT ids
            print "Progress: ",(prog * 100) / total, "%"
        results.append(Result(np.mean(maps), np.std(maps), pars))
        print maps
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
                    # print relevant_u, hidden_ratings[u], URM_test[u,].nonzero()[1]
            maps.append(map_scorer(rec, URM_train[test_rows,], hidden_ratings, n))  # Assume rec to predict indices of items, NOT ids
            print "Progress: ", (prog * 100) / total, "%"
            prog += 1
        results.append(Result(np.mean(maps), np.std(maps), pars))
        print maps
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



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def read_items():
    items_df = pd.read_csv('../../inputs/item_profile.csv', sep='\t')
    # items_df.career_level = items_df.career_level.fillna(0).astype('int64')
    return items_df


def read_interactions():
    ints = pd.read_csv('../../inputs/interactions_idx.csv', sep='\t')
    return sps.csr_matrix((ints['interaction_type'].values, (ints['user_idx'].values, ints['item_idx'].values)))


def read_top_pops():
    aux = pd.read_csv('../../inputs/top_pop_ids.csv', sep='\t')['0'].values
    # print aux
    return aux


def cross_validate(items_df, URM, n):
    """
    Arguments:
    items_df -- ICM DataFrame
    URM
    n -- number of recommendations
    """

    params = {'k': [1, 5, 10], 'sh': [0, 2]}
    # params = {'k':[1], 'sh':[0]}
    rec = GridSearchCV(ItemCB(), params, scoring=map_scorer, cv=2, fit_params={'n': n, 'items_df': items_df})

    #  id_dic = {item_df.loc[i]['id']:i for i in range(item_df.shape[0])}
    tup = URM.nonzero()
    item_ids = map(lambda x: items_df.loc[x]['id'], tup[1])
    # print item_ids
    n_users = len(item_ids)
    pos_items = [[] for _ in range(URM.shape[0])]
    for i in range(n_users):
        pos_items[tup[0][i]].append(item_ids[i])

    rec.fit(URM, pos_items)
    scores = pd.DataFrame(
        data=[[_.mean_validation_score, np.std(_.cv_validation_scores)] + _.parameters.values() for _ in
              rec.grid_scores_],
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


class ItemCB(BaseEstimator):
    def __init__(self, k=5, sh=2, k_top=500): #k_rated??
        self.k = k
        self.sh = sh
        self.k_top = k_top
        self.pop = None
        self.ICM = None

    # Train recommender
    def fit(self, URM, ICM, actives):
        """
        Arguments:
        n -- number of recommendations
        pos_items -- list of item indices rated by users (used by GridSeahCV)
        URM -- user rating matrix
        """
        st = time.time()
        self.ICM = sps.csr_matrix(ICM) # This makes a reference to ICM, not a copy

        item_pop = URM.sum(axis=0)  # Count or sum??
        item_pop = np.asarray(item_pop).squeeze()
        self.pop = np.argsort(item_pop)[::-1]
        mask = np.array([True if actives[i] == 1 else False for i in self.pop])
        self.pop = self.pop[mask][:self.k_top]
        # pd.DataFrame(self.pop).to_csv('../../inputs/top_pop_ids.csv', sep='\t', index=False) #Saving ids and not indices!!
        # self.pop = read_top_pops()[:62]
        et = time.time()
        #print "Fit time: ", et - st

    # Make predictions on trained recommender
    # Returns preidctions matrix with indices
    def predict(self, URM, n):
        recs = [[] for _ in range(URM.shape[0])] #TODO: Optimize inside for
        for u in range(URM.shape[0]):
            st = time.time()
            rated = URM[u].nonzero()[1]
            # print "User", u, len(rated)
            if len(rated) == 0:
                recs[u] = list(self.pop[:n])  # [self.item_ids[i] for i in self.pop[:self.n]]
            else:
                for i in self.pop:
                    if URM[u, i] == 0:
                        closest = self.calculate_knn(i, rated)
                        den = 0
                        rating = 0
                        for j in closest:
                            rating += URM[u, j[1]] * j[0]
                            den += j[0]
                        rating /= 1 if den == 0 else den

                        if len(recs[u]) < n:
                            heapq.heappush(recs[u], (rating, i))
                        else:
                            if rating > recs[u][0][0]:
                                heapq.heappushpop(recs[u], (rating, i))
                                # print Y[u], u, len(rated)
                recs[u].sort(reverse=True) #We were puttinh the in incresing order!!
                # print Y[u], u, len(rated)
                recs[u] = [i[1] for i in recs[u]]  # [self.item_ids[i[1]] for i in Y[u]]
            # print Y[u], u, len(rated)
            assert len(recs[u]) > 0
            et = time.time()
            #print "User rec time", et - st
        return recs

    def compute_similarity_matrix(self, norm_ICM):
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
        denominator_matrix = norm2_values.dot(norm2_values.transpose()).applymap(lambda x: 1 / (x + self.sh))

        # Check for square matrix
        if (denominator_matrix.values.shape[0] != denominator_matrix.value.shape[1]):
            logger.error('The resulting similarity matrix is not square!')

        return numerator_matrix.dot(denominator_matrix.transpose())

    def sim(self, i, j):
        v_i, v_j = self.ICM[i,], self.ICM[j,]
        return (v_i.dot(v_j.transpose()) / (linalg.norm(v_i) * linalg.norm(v_j) + self.sh))[0, 0]
        # res = 0
        # # num_atts = self.attr_df.shape[1] + reduce(lambda _, mat:_+mat.shape[1], self.attr_mat.itervalues(), 0)
        # # m_start_time = time.time()
        # v_i, v_j = sps.csr_matrix(self.attr_df.iloc[[i]].values), sps.csr_matrix(self.attr_df.iloc[[j]].values)
        # # print num_atts
        #
        # for _, v in self.attr_mat.items():
        #     v_i = sps.hstack([v_i, v[i]])
        #     v_j = sps.hstack([v_j, v[j]])
        #     # print _, v.shape[1]
        # # m_end_time = time.time()
        # # print "matrix creation", m_end_time - m_start_time
        #
        # # d_start_time = time.time()
        # aux = (v_i.dot(v_j.transpose()) / (linalg.norm(v_i) * linalg.norm(v_j) + self.sh))[0, 0]
        # # d_end_time = time.time()
        #
        # # print "dot prod", d_end_time - d_start_time
        # # print "total time", d_end_time - d_start_time + m_end_time - m_start_time
        #
        # # st = time.time()
        # # num = 0.0
        # # sqr_sum_i = 0.0
        # # sqr_sum_j = 0.0
        # # v_i, v_j = sps.csr_matrix(self.attr_df.iloc[[i]].values), sps.csr_matrix(self.attr_df.iloc[[j]].values)
        # # for _, v in self.attr_mat.items():
        # #    v_j_t = v[j].transpose()
        # #    num += v[i].dot(v_j_t)
        # #    sqr_sum_i += v[i].dot(v[i].transpose())
        # #    sqr_sum_j += v[j].dot(v_j_t)
        # #
        # # v_j_t = v_j.transpose()
        # # num += v_i.dot(v_j_t)
        # # sqr_sum_i += v_i.dot(v_i.transpose())
        # # sqr_sum_j += v_j.dot(v_j_t)
        # #
        # #
        # #
        # # aux2 = num[0,0] / (math.sqrt(sqr_sum_i[0,0] * sqr_sum_j[0,0]) + self.sh)
        # # et = time.time()
        # # print "Alt total time", et - st
        #
        # # stttt = time.time()
        # # num = 0.0
        # # sqr_sum_i = 0.0
        # # sqr_sum_j = 0.0
        # # for k in range(self.attr_df.shape[1]):
        # #    aik = self.attr_df.iloc[[i], [k]].values[0][0]
        # #    ajk = self.attr_df.iloc[[j], [k]].values[0][0]
        # #    num += aik * ajk
        # #    sqr_sum_i += aik * aik
        # #    sqr_sum_j += ajk * ajk
        # #
        # #    for _, v in self.attr_mat.items():
        # #
        # #       for k in range(v.shape[1]):
        # #
        # #          aik = v[i,k]
        # #          ajk = v[j,k]
        # #          num += aik * ajk
        # #          sqr_sum_i += aik * aik
        # #          sqr_sum_j += ajk * ajk
        # # etttt = time.time()
        # # time3 = etttt - stttt
        # # print "Alt 2 total time", time3
        # return aux

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
    aux = pd.get_dummies(attr_df, columns=to_dummies)
    print aux.columns, aux.columns.shape
    trans = CountVectorizer(token_pattern='\w+')
    for attr in to_dummies:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()
        print attr, attr_mats[attr].shape, trans.vocabulary_

    # attr_mats = {_: generate_tfidf(attr_df[_].map(str).values) for _ in to_tfidf}
    trans = TfidfVectorizer(token_pattern='\w+')
    for attr in to_tfidf:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()
        print attr, attr_mats[attr].shape, trans.vocabulary_

    logger.info(attr_df.columns)
    return reduce(lambda acc, x: x if acc.shape == (1, 1) else sps.hstack([acc, x]), attr_mats.itervalues(),
                  sps.lil_matrix((1, 1)))


# Necessary?
def generate_tfidf(self, data):
    vectorizer = CountVectorizer(token_pattern='\w+')
    trans = TfidfTransformer()
    tf = vectorizer.fit_transform(data)
    return trans.fit_transform(tf)  # , vectorizer.vocabulary_)



items_df = read_items()
URM = read_interactions()
# test_users = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values[:10]
# urm = urm[test_users,]
actives = items_df.active_during_test.values
item_ids = items_df.id.values
ICM = generate_attrs(items_df)
rec = ItemCB(k_top=62)
#CVSearch(rec, URM, ICM, actives, 4000)
HoldoutSearch(rec, URM, ICM, actives, 40000)
# a.fit(urm, None, 5, items_df)
# recs = a.predict(urm)
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
# print "CV", et-st