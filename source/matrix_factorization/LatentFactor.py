import numpy as np
import scipy.sparse as sps
from collections import namedtuple
from sklearn.model_selection import KFold, ParameterGrid
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import time
import pandas as pd
import sys

sys.path.append('./../')
import utils.utils as ut
from TopPopular.TopPopular import TopPop


def cv_search(rec, urm, non_active_items_mask, sample_size, sample_from_urm=True):
    np.random.seed(1)
    urm_sample, icm_sample, _, non_active_items_mask_sample = ut.produce_sample(urm, icm=None, ucm=None,
                                                                                non_active_items_mask=non_active_items_mask,
                                                                                sample_size=sample_size,
                                                                                sample_from_urm=sample_from_urm)
    params = {'k':[10, 100, 1000],'reg_penalty':[0.0001, 0.001, 0.01, 0.1, 1, 10]}
    params = {'k':[1000]}
    grid = list(ParameterGrid(params))
    folds = 4
    kfold = KFold(n_splits=folds)
    splits = [(train, test) for train, test in kfold.split(urm_sample)]
    retained_ratings_perc = 0.75
    n = 5
    result = namedtuple('result', ['mean_score', 'std_dev', 'parameters'])
    results = []
    total = float(reduce(lambda acc, x: acc * len(x), params.itervalues(), 1) * folds)
    prog = 1.0

    for pars in grid:
        print pars
        rec = rec.set_params(**pars)
        # rec.l1_ratio = rec.l1_penalty / (rec.l1_penalty + rec.l2_penalty)
        # rec.top_pop.count = pars['count_top_pop']
        maps = []

        for row_train, row_test in splits:
            urm_train = urm_sample[row_train, :]
            urm_test = urm_sample[row_test, :]
            hidden_ratings = []
            for u in range(urm_test.shape[0]):
                relevant_u = urm_test[u,].nonzero()[1]  # Indices of rated items for test user u
                if len(relevant_u) > 1:  # 1 or 2
                    np.random.shuffle(relevant_u)
                    urm_test[u, relevant_u[int(len(relevant_u) * retained_ratings_perc):]] = 0
                    hidden_ratings.append(relevant_u[int(len(relevant_u) * retained_ratings_perc):])
                else:
                    hidden_ratings.append([])
            urm_test.eliminate_zeros()
            rec.fit(urm_test)
            maps.append(ut.map_scorer(rec, np.arange(urm_test.shape[0]), hidden_ratings, n,
                                      non_active_items_mask_sample))  # Assume rec to predict indices of items, NOT ids
            print "Progress: {:.2f}%".format((prog * 100) / total)
            prog += 1
        print maps
        results.append(result(np.mean(maps), np.std(maps), pars))
        print "Result: ", result(np.mean(maps), np.std(maps), pars)
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP", "Std"] + _.parameters.keys())
    print "Total scores: ", scores
    scores.to_csv('LatentFactor CV MAP values 1.csv', sep='\t', index=False)
    '''cols, col_feat, x_feat = 3, 'l2_penalty', 'l1_penalty'
    f = sns.FacetGrid(data=scores, col=col_feat, col_wrap=cols, sharex=False, sharey=False)
    f.map(plt.plot, x_feat, 'MAP')
    f.fig.suptitle("SLIM-Top pop CV MAP values")
    i_max, y_max = scores['MAP'].argmax(), scores['MAP'].max()
    i_feat_max = params[col_feat].index(scores[col_feat][i_max])
    f_max = f.axes[i_feat_max]
    f_max.plot(scores[x_feat][i_max], y_max, 'o', color='r')
    plt.figtext(0, 0, "With 500 top pops\nMaximum at (sh={:.5f},k={:.5f}, {:.5f}+/-{:.5f})".format(
        scores[col_feat][i_max],
        scores[x_feat][i_max],
        y_max,
        scores['Std'][i_max]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    f.savefig('SLIM_Item CV MAP values 1.png', bbox_inches='tight')'''


class LatentFactor(BaseEstimator):
    def __init__(self, k, max_steps=5000, min_error=0.001, min_delta_error = 0.001, learn_rate=0.001, reg_penalty=0.02, seed=1, pred_batch_size=2500):
        self.k = k
        self.max_steps = max_steps
        self.min_error = min_error
        self.min_delta_error = min_delta_error
        self.learn_rate = learn_rate
        self.reg_penalty = reg_penalty
        self.seed = seed
        self.P = None
        self.Q = None
        self.R = None
        self.pred_batch_size = pred_batch_size

    def fit(self, R):
        np.random.seed(self.seed)
        self.P = np.random.randn(R.shape[0], self.k+2)
        self.Q = np.random.randn(R.shape[1], self.k+2)
        self.P[:, -1] = np.ones(self.P.shape[0])
        self.Q[:, -2] = np.ones(self.Q.shape[0])
        self.R = R
        step = 1
        delta_err = 1
        error = 1

        while step <= self.max_steps and delta_err > self.min_delta_error and error > self.min_error:
            st = time.time()
            E = sps.csr_matrix(R, copy=True, dtype=self.P.dtype)
            et = time.time()
            print et-st
            st = time.time()
            tup = E.nonzero()
            rows, cols = tup[0], tup[1]
            for i in range(E.data.shape[0]):
                E[rows[i], cols[i]] -= self.P[rows[i], :].dot(self.Q[cols[i], :])
            et = time.time()
            print et - st
            st = time.time()
            P_prime = (1 - self.learn_rate * self.reg_penalty) * self.P + self.learn_rate * E.dot(self.Q)
            Q_prime = (1 - self.learn_rate * self.reg_penalty) * self.Q + self.learn_rate * E.T.dot(self.P)
            et = time.time()
            print et - st
            st = time.time()
            self.P, self.Q = P_prime.copy(), Q_prime.copy()
            self.P[:, -1] = np.ones(self.P.shape[0])
            self.Q[:, -2] = np.ones(self.Q.shape[0])
            del P_prime, Q_prime
            et = time.time()
            print et - st
            st = time.time()
            error = np.sum(E.data ** 2) / 2 + self.reg_penalty * np.sum(self.P ** 2) / 2 + self.reg_penalty * np.sum(self.Q ** 2) / 2
            if step > 1:
                delta_err = abs(prev_error - error)
            print "Step ", step, ", error ", error, ", delta error ", delta_err
            step += 1
            prev_error = error
            et = time.time()
            print et-st

    def predict(self,user_rows,n,non_active_items_mask):
        n_iterations = user_rows.shape[0] / self.pred_batch_size + (user_rows.shape[0] % self.pred_batch_size != 0)
        ranking = None

        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations

            start = i * self.pred_batch_size
            end = start + self.pred_batch_size if i < n_iterations - 1 else user_rows.shape[0]
            batch_rows = user_rows[start:end]
            batch_scores = self.P[batch_rows,:].dot(self.Q.T)
            nonzero_indices = self.R[batch_rows,:].nonzero()
            batch_scores[nonzero_indices[0], nonzero_indices[1]] = 0.0

            # remove the inactives items
            batch_scores[:, non_active_items_mask] = 0.0
            batch_ranking = batch_scores.argsort()[:, ::-1]
            batch_ranking = batch_ranking[:, :n]  # leave only the top n

            if i == 0:
                ranking = batch_ranking.copy()
            else:
                ranking = np.vstack((ranking, batch_ranking))
        return ranking

R = sps.csr_matrix(ut.read_interactions(), copy=False, dtype=np.float64)
# global_bias = np.mean(R.data)
# R.data -= global_bias
items_dataframe = ut.read_items()
item_ids = items_dataframe.id.values
actives = np.array(items_dataframe.active_during_test.values)
non_active_items_mask = actives == 0
rec = LatentFactor(k=20, learn_rate=0.001, max_steps=1000, pred_batch_size=1000)
# cv_search(rec, R, non_active_items_mask, sample_size=None,sample_from_urm=True)

test_users_idx = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
rec = LatentFactor(k=20, learn_rate=0.001, max_steps=2000, pred_batch_size=1000, reg_penalty=0.001)
rec.fit(R)
ranking = rec.predict(user_rows=test_users_idx, n=5, non_active_items_mask=non_active_items_mask)
ut.write_recommendations("LatentFactor k20 steps1000 reg10minus3", ranking, test_users_idx, item_ids)

rec = LatentFactor(k=20, learn_rate=0.001, max_steps=2000, pred_batch_size=1000, reg_penalty=0.01)
rec.fit(R)
ranking = rec.predict(user_rows=test_users_idx, n=5, non_active_items_mask=non_active_items_mask)
ut.write_recommendations("LatentFactor k20 steps1000 reg10minus2", ranking, test_users_idx, item_ids)

rec = LatentFactor(k=20, learn_rate=0.001, max_steps=2000, pred_batch_size=1000, reg_penalty=0.1)
rec.fit(R)
ranking = rec.predict(user_rows=test_users_idx, n=5, non_active_items_mask=non_active_items_mask)
ut.write_recommendations("LatentFactor k20 steps1000 reg10minus1", ranking, test_users_idx, item_ids)