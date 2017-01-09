import numpy as np
import scipy.sparse as sps
from collections import namedtuple
from sklearn.model_selection import KFold, ParameterGrid
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF, TruncatedSVD
import time
import pandas as pd
import sys

sys.path.append('./../')
sys.path.append('./')
import utils.utils as ut
import MF_utils


def cv_search(rec, urm, non_active_items_mask, sample_size, sample_from_urm=True):
    np.random.seed(1)
    urm_sample, icm_sample, _, non_active_items_mask_sample = ut.produce_sample(urm, icm=None, ucm=None,
                                                                                non_active_items_mask=non_active_items_mask,
                                                                                sample_size=sample_size,
                                                                                sample_from_urm=sample_from_urm)
    params = {'k':[10, 20, 50, 100, 150, 200],'reg_penalty':[0.0001, 0.001, 0.01, 0.1, 1, 10]}
    params = {'k': [10, 20, 50, 100, 150, 200, 500], 'reg_penalty': [10, 20, 50, 100, 200, 500, 1000]}
    params = {'k':[50, 100, 200], 'reg_penalty':[100, 10000, 1000000, 1e8]}
    params = {'k':[50], 'reg_penalty': [10]}
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
        maps = []
        '''if pars['fit_user_bias'] and pars['fit_item_bias']:
            continue'''

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
            print maps
            prog += 1
            break
        results.append(result(np.mean(maps), np.std(maps), pars))
        print "Result: ", result(np.mean(maps), np.std(maps), pars)
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP", "Std"] + _.parameters.keys())
    print "Total scores: ", scores
    #scores.to_csv('LatentFactor CV MAP values 5.csv', sep='\t', index=False)
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
    def __init__(self, k, fit_user_bias=False, fit_item_bias=False, max_steps=5000, min_error=0.001, min_grad_norm = 0.001,
                 min_delta_err=0.0001,learn_rate=0.001, reg_penalty=10, l1_ratio=0, seed=1, pred_batch_size=2500):
        self.k = k
        self.fit_user_bias = fit_user_bias
        self.fit_item_bias = fit_item_bias
        self.max_steps = max_steps
        self.min_error = min_error
        self.min_grad_norm = min_grad_norm
        self.min_delta_err = min_delta_err
        self.learn_rate = learn_rate
        self.reg_penalty = reg_penalty
        self.l1_ratio = l1_ratio
        self.seed = seed
        self.P = None
        self.Q = None
        self.R = None
        self.pred_batch_size = pred_batch_size


    def fit(self, R):
        '''nmf = NMF(n_components=self.k, alpha=self.reg_penalty, l1_ratio=self.l1_ratio, verbose=1)
        self.P = nmf.fit_transform(R)
        self.Q = nmf.components_.T
        self.R = R
        return'''

        np.random.seed(self.seed)
        self.P = np.random.randn(R.shape[0], self.k+self.fit_user_bias + self.fit_item_bias)
        self.Q = np.random.randn(R.shape[1], self.k+self.fit_user_bias + self.fit_item_bias)
        #print np.mean(self.P), np.std(self.P), np.mean(self.Q), np.std(self.Q)
        if self.fit_user_bias:
            self.Q[:, -2 + (not self.fit_item_bias)] = np.ones(self.Q.shape[0])
        if self.fit_item_bias:
            self.P[:, -1] = np.ones(self.P.shape[0])
        #print np.mean(self.P), np.std(self.P), np.mean(self.Q), np.std(self.Q)
        self.R = R
        step = 1
        grad_norm = 1
        error = 1
        delta_err = 1

        while step <= self.max_steps and grad_norm > self.min_grad_norm and error > self.min_error and delta_err > self.min_delta_err:
            st = time.time()
            E = MF_utils.calculate_error_matrix(R, self.P, self.Q)
            e_error, p_error, q_error = np.sum(E.data ** 2) / 2, self.reg_penalty * np.sum(self.P** 2) / 2, self.reg_penalty * np.sum(self.Q ** 2) / 2
            if self.fit_user_bias:
                q_error -= self.reg_penalty*np.sum(self.Q[:, -2 + (not self.fit_item_bias)]**2) / 2
            if self.fit_item_bias:
                p_error -= self.reg_penalty*np.sum(self.P[:, -1]**2) / 2
            error = e_error + p_error + q_error
            # error /= self.reg_penalty

            e_dot_q = E.dot(self.Q)
            e_dot_p = E.T.dot(self.P)

            grad_P = -e_dot_q + self.reg_penalty * self.P
            grad_Q = -e_dot_p + self.reg_penalty * self.Q
            if self.fit_user_bias:
                grad_Q[:,-2 + (not self.fit_item_bias)] = 0.0
            if self.fit_item_bias:
                grad_P[:,-1] = 0.0
            # grad_P /= self.reg_penalty
            # grad_Q /= self.reg_penalty
            grad_norm = np.sqrt(np.sum(grad_P** 2) + np.sum(grad_Q**2))

            '''ints = R.nonzero()
            for i in np.random.permutation(R.data.shape[0])-1:
                u_i = self.P[ints[0][i],:]
                v_j = self.Q[ints[1][i],:]
                e_ij = R.data[i] - u_i.dot(v_j)
                self.P[ints[0][i], :] = u_i + self.learn_rate*(e_ij*v_j - self.reg_penalty*u_i)
                self.Q[ints[1][i], :] = v_j + self.learn_rate*(e_ij*u_i - self.reg_penalty*v_j)
                if self.fit_user_bias:
                    self.Q[ints[1][i], -2 + (not self.fit_item_bias)] = 1
                if self.fit_item_bias:
                    self.P[ints[0][i],-1] = 1'''
            P_prime = (1 - self.learn_rate * self.reg_penalty) * self.P + self.learn_rate * e_dot_q
            Q_prime = (1 - self.learn_rate * self.reg_penalty) * self.Q + self.learn_rate * e_dot_p
            if self.fit_user_bias:
                Q_prime[:, -2 + (not self.fit_item_bias)] = np.ones(self.Q.shape[0])
            if self.fit_item_bias:
                P_prime[:, -1] = np.ones(self.P.shape[0])
            #print np.mean(np.mean(self.P, axis=1)), np.std(np.mean(self.P, axis=1)), np.mean(np.mean(self.Q, axis=1)), np.std(np.mean(self.Q, axis=1))
            #print np.mean(self.P), np.std(self.P), np.mean(self.Q), np.std(self.Q)
            #print np.mean(self.P[1, :]), np.std(self.P[1, :]), np.mean(self.Q[1, :]), np.std(self.Q[1, :])
            #print np.mean(self.P[2, :]), np.std(self.P[2, :]), np.mean(self.Q[2, :]), np.std(self.Q[2, :])
            self.P, self.Q = P_prime.copy(), Q_prime.copy()
            del P_prime, Q_prime

            if step > 1:
                delta_err = prev_error - error
            if step % 100 == 0:
                print "Step ", step, ", error ", error, ", delta error ", delta_err, ", grad norm ", grad_norm

            step += 1
            prev_error = error
            et = time.time()
            # print et-st
        print "Step ", step, ", error ", error, ", delta error ", delta_err, ", grad norm ", grad_norm
        '''print np.mean(self.P[0,:]), np.std(self.P[0,:]), np.mean(self.Q[0,:]), np.std(self.Q[0,:])
        print np.mean(self.P[1, :]), np.std(self.P[1, :]), np.mean(self.Q[1, :]), np.std(self.Q[1, :])
        print np.mean(self.P[2, :]), np.std(self.P[2, :]), np.mean(self.Q[2, :]), np.std(self.Q[2, :])
        print self.P[0, :5], self.P[0, :5], self.Q[0, :5], self.Q[0, :5]'''

    def predict(self,user_rows,n,non_active_items_mask):
        n_iterations = user_rows.shape[0] / self.pred_batch_size + (user_rows.shape[0] % self.pred_batch_size != 0)
        ranking = None

        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations

            start = i * self.pred_batch_size
            end = start + self.pred_batch_size if i < n_iterations - 1 else user_rows.shape[0]
            batch_rows = user_rows[start:end]
            batch_scores = self.P[batch_rows,:].dot(self.Q.T)# + global_bias
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
R, global_bias, item_bias, user_bias = ut.global_effects(R)
#R.data -= global_bias
items_dataframe = ut.read_items()
item_ids = items_dataframe.id.values
actives = np.array(items_dataframe.active_during_test.values)
non_active_items_mask = actives == 0
rec = LatentFactor(k=200, fit_item_bias=False, fit_user_bias=False, learn_rate=0.001, min_delta_err=0.001,
                   min_grad_norm=0.001, max_steps=5000, pred_batch_size=1000, reg_penalty=1, l1_ratio=0)
#rec.fit(R)
cv_search(rec, R, non_active_items_mask, sample_size=10000,sample_from_urm=True)
'''svd = TruncatedSVD(n_components=500)
svd.fit_transform(R)
exp_var=0
exp_rat=0
for i in range(svd.n_components):
    exp_rat += svd.explained_variance_ratio_[i]
    exp_var += svd.explained_variance_[i]
    print i, svd.explained_variance_ratio_[i], exp_rat, svd.explained_variance_[i], exp_var'''

'''test_users_idx = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
rec.fit(R)
ranking = rec.predict(user_rows=test_users_idx, n=5, non_active_items_mask=non_active_items_mask)
ut.write_recommendations("LatentFactor500", ranking, test_users_idx, item_ids)'''
