import random
import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import time
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid
from collections import namedtuple



def map_scorer(rec, URM, items_ids):
   score = 0
   rec_list = rec.predict(URM)
   i = 0
   for u in range(URM.shape[0]):
      if len(items_ids[u]) > 0 :
         is_relevant = np.in1d(rec_list[u], items_ids[u], assume_unique=True)
         p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
         # print u, rec_list[u], len(rec_list[u]), len(rec_list), URM.shape
         # assert len(rec_list[u]) > 0
         score += np.sum(p_at_k) / np.min([len(items_ids[u]), len(rec_list[u])])
         i += 1
   return score / i


def CVSearch(rec, URM, ICM, actives, sample_size=None):
    np.random.seed(1)
    URM_sample = URM[np.random.permutation(URM.shape[0])[:sample_size],]
    params = {'k':[5],'sh':[2]}
    grid = list(ParameterGrid(params))
    folds = 4
    kfold = KFold(n_splits=folds, shuffle=True)  # Suffle UCM if necessary too, to keep indices correspondence
    splits = kfold.split(URM_sample)
    Result = namedtuple('Result', ['mean_score', 'std_dev', 'parameters'])
    results = []
    pred_ratings_perc = 0.75
    n = 5

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
    train_size = 0.75
    params = {'k': [5], 'sh': [2]}
    grid = list(ParameterGrid(params))
    repetitions = 4
    Result = namedtuple('Result', ['mean_score', 'std_dev', 'parameters'])
    results = []
    n = 5

    for pars in grid:
        rec = rec.set_params(**pars)
        maps = []
        for rep in range(repetitions):
            np.random.shuffle(perm)
            i_train, i_test = perm[:int(len(perm) * train_size)], perm[int(len(perm) * train_size):]
            URM_train = sps.csr_matrix((unique_rows.size, URM.shape[1]))
            for i in range(len(i_train)):
                URM_train[nnz_to_sample_row[nnz[0][i]], nnz[1][i]] = URM[nnz[0][i], nnz[1][i]]
            print URM_train.count_nonzero(), URM.count_nonzero()
            rec.fit(URM_train, ICM, actives)
            test_rows = np.unique([nnz_to_sample_row[nnz[0][i]] for i in i_test])
            URM_test = sps.csr_matrix((unique_rows.size, URM.shape[1]))
            URM_test[test_rows,] = URM_test[test_rows,]
            hidden_ratings = []
            for u in range(URM_test.shape[0]):
                relevant_u = URM_test[u,].nonzero()[1]  # Indices of rated items for test user u
                if len(relevant_u) > 0:
                    hidden_ratings.append(relevant_u)
                    # print relevant_u, hidden_ratings[u], URM_test[u,].nonzero()[1]
                else:
                    hidden_ratings.append([])
            maps.append(map_scorer(rec, URM_test, hidden_ratings, n))  # Assume rec to predict indices of items, NOT ids
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