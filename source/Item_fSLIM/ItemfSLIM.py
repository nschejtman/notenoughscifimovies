import numpy as np
import scipy.sparse as sps
from collections import namedtuple
from sklearn.model_selection import KFold, ParameterGrid
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.base import BaseEstimator
import time
import pandas as pd
import sys
sys.path.append('./../')
import utils.utils as ut
from TopPopular.TopPopular import TopPop
from multiprocessing import Pool
from functools import partial


def cv_search(rec, urm, icm, non_active_items_mask, sample_size, sample_from_urm=True):
    np.random.seed(1)
    urm_sample, icm_sample, _, non_active_items_mask_sample = ut.produce_sample(urm, icm=icm, ucm=None,
                                                                                 non_active_items_mask=non_active_items_mask,
                                                                                 sample_size=sample_size, sample_from_urm=sample_from_urm)
    params = {'l1_ratio':[1e-4, 1e-5, 1e-6, 1e-7], 'k_nn':[2000, 3000, 4000, 5000, 6000],
              'aa_sh':[1000, 2000, 5000]}
    grid = list(ParameterGrid(params))
    folds = 4
    kfold = KFold(n_splits=folds)
    splits = [(train, test) for train,test in kfold.split(urm_sample)]
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

        for row_train, row_test in splits:
            urm_train = urm_sample[row_train,:]
            rec.fit(urm_train, icm_sample)
            urm_test = urm_sample[row_test,:]
            hidden_ratings = []
            for u in range(urm_test.shape[0]):
                relevant_u = urm_test[u,].nonzero()[1]  # Indices of rated items for test user u
                if len(relevant_u) > 1:#1 or 2
                    np.random.shuffle(relevant_u)
                    urm_test[u, relevant_u[int(len(relevant_u) * retained_ratings_perc):]] = 0
                    hidden_ratings.append(relevant_u[int(len(relevant_u) * retained_ratings_perc):])
                else:
                    hidden_ratings.append([])
            maps.append(ut.map_scorer(rec, urm_test, hidden_ratings, n, non_active_items_mask_sample))  # Assume rec to predict indices of items, NOT ids
            print "Progress: {:.2f}%".format((prog * 100) / total)
            prog += 1
        print maps
        results.append(result(np.mean(maps), np.std(maps), pars))
        print "Result: ", result(np.mean(maps), np.std(maps), pars)
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP", "Std"] + _.parameters.keys())
    print "Total scores: ", scores
    scores.to_csv('fSLIM_Item CV MAP values 1.csv', sep='\t', index=False)
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



class SLIM_recommender(BaseEstimator):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self, top_pops, l1_ratio=1e-4, positive_only=True, alpha_ridge=None, alpha_lasso=None,
                 k_nn=10, aa_sh=2000, sim_partition_size=2500, pred_batch_size=2500):
        super(SLIM_recommender, self).__init__()
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.alpha_ridge = alpha_ridge
        self.alpha_lasso = alpha_lasso
        self.k_nn = k_nn
        self.sh = aa_sh
        self.sim_partition_size = sim_partition_size
        self.top_pops = top_pops
        self.pred_batch_size = pred_batch_size


    def fit(self, URM, ICM):
        print time.time(), ": ", "Started fit"
        URM = ut.check_matrix(URM, 'csc', dtype=np.float32)
        n_items = URM.shape[1]
        values, rows, cols = [], [], []

        if self.alpha_ridge is not None:
            self.model = Ridge(self.alpha_ridge, copy_X=False, fit_intercept=False)
        elif self.alpha_lasso is not None:
            self.model = Lasso(alpha=self.alpha_lasso, copy_X=False, fit_intercept=False)
        else:
            self.model = ElasticNet(alpha=1.0, l1_ratio=self.l1_ratio, positive=self.positive_only, fit_intercept=False, copy_X=False)

        k = 1

        for j in np.sort(self.top_pops):#, because only the active ones are to be recommended(range(n_items) if self.k_top is None else top_pops):
            top_k_idx = get_knn(ICM, j, self.k_nn, self.sh)
            y = URM[:, j].toarray()

            if self.alpha_ridge is None and self.alpha_lasso is None:
                self.model.fit(URM[:,top_k_idx], y)
            else:
                self.model.fit(URM[:,top_k_idx], y.ravel())

            nnz_mask = self.model.coef_ > 0.0
            values.extend(self.model.coef_[nnz_mask])
            rows.extend(top_k_idx[nnz_mask])
            cols.extend(np.ones(nnz_mask.sum()) * j)

            if k % 1000 == 0:
                print time.time(), k

            k += 1

        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        print time.time(), ": ", "Finished fit"


    def predict(self, URM, n_of_recommendations=5, non_active_items_mask=None):
        print time.time(), ": ", "Started predict"
        user_profile = URM
        n_iterations = user_profile.shape[0] / self.pred_batch_size + (user_profile.shape[0] % self.pred_batch_size != 0)
        ranking = None

        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations
            start = i * self.pred_batch_size
            end = start + self.pred_batch_size if i < n_iterations - 1 else user_profile.shape[0]

            batch_profiles = URM[start:end,:]
            batch_scores = batch_profiles.dot(self.W_sparse).toarray().astype(np.float32)

            nonzero_indices = batch_profiles.nonzero()
            batch_scores[nonzero_indices[0], nonzero_indices[1]] = 0.0

            # remove the inactives items
            batch_scores[:, non_active_items_mask] = 0.0
            batch_ranking = batch_scores.argsort()[:, ::-1]
            batch_ranking = batch_ranking[:, :n_of_recommendations]  # leave only the top n

            sum_of_scores = batch_scores[np.arange(batch_scores.shape[0]), batch_ranking.T].T.sum(axis=1).ravel()
            zero_scores_mask = sum_of_scores == 0
            n_zero_scores = np.extract(zero_scores_mask, sum_of_scores).shape[0]
            if n_zero_scores != 0:
                batch_ranking[zero_scores_mask] = [self.top_pops[:n_of_recommendations] for _ in range(n_zero_scores)]

            if i == 0:
                ranking = batch_ranking.copy()
            else:
                ranking = np.vstack((ranking, batch_ranking))

        print time.time(), ": ", "Finished predict"

        return ranking


def get_knn(icm, i, k, sh):
    #icm = ut.normalize_matrix(icm, row_wise=True)
    sims = icm[i,:].dot(icm.T).toarray().ravel()
    sims[i] = 0.0
    icm_ind = icm.copy()
    icm_ind.data = np.ones_like(icm_ind.data)
    counts = icm_ind[i,:].dot(icm_ind.T).toarray().ravel()
    counts /= (counts + sh)
    sims *= counts
    top_k = np.argsort(sims).ravel()
    return top_k[-k:]


def calculate_and_save_similarities(max_k, shs, partition_size):
    items_dataframe = ut.read_items()
    icm = ut.generate_icm(items_dataframe)
    for sh in shs:
        sim, top_k_idx = ut.compute_similarity_matrix_knn(icm, max_k, sh, row_wise=True, partition_size=partition_size)
        np.save('Similarity'+str(max_k) + '_' + str(sh)+'data', sim.data)
        np.save('Similarity' + str(max_k) + '_' + str(sh)+'indptr', sim.indptr)
        np.save('Similarity' + str(max_k) + '_' + str(sh)+'indices', sim.indices)
        np.save('Similarity' + str(max_k) + '_' + str(sh)+'shape', sim.shape)
        np.save('Similarity' + str(max_k) + '_' + str(sh)+'top_k_idx', top_k_idx)


def load_similarities(max_k, sh):
    sim = sps.csr_matrix((np.load('Similarity' + str(max_k) + '_' + str(sh) + 'data.npy'),
                          np.load('Similarity' + str(max_k) + '_' + str(sh) + 'indices.npy'),
                          np.load('Similarity' + str(max_k) + '_' + str(sh) + 'indptr.npy')),
                         np.load('Similarity' + str(max_k) + '_' + str(sh) + 'shape.npy'))
    top_k_idx = np.load('Similarity' + str(max_k) + '_' + str(sh) + 'top_k_idx.npy')
    return sim, top_k_idx


urm = ut.read_interactions()
items_dataframe = ut.read_items()
icm = ut.generate_icm(items_dataframe)
icm = ut.normalize_matrix(icm, row_wise=True)
item_ids = items_dataframe.id.values
actives = np.array(items_dataframe.active_during_test.values)
non_active_items_mask = actives == 0
test_users_idx = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
urm_pred = urm[test_users_idx, :]

top_rec = TopPop(count=True)
top_rec.fit(urm)
top_pops = top_rec.top_pop[non_active_items_mask[top_rec.top_pop] == False]

# TODO: Use all top_pops or only active ones in fitting??
recommender = SLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000)
# recommender.fit(urm)
# cv_search(recommender, urm, icm, non_active_items_mask, sample_size=10000, sample_from_urm=True)

params = {'l1_ratio':[1e-4, 1e-5, 1e-6, 1e-7], 'k_nn':[2000, 3000, 4000, 5000, 6000], 'aa_sh':[1000, 2000, 5000]}
grid = list(ParameterGrid(params))

for pars in grid:
    print str(pars)
    recommender = recommender.set_params(**pars)
    recommender.fit(urm, icm)
    ranking = recommender.predict(urm_pred, non_active_items_mask=non_active_items_mask, n_of_recommendations=5)
    ut.write_recommendations("Item fSLIM "+str(pars), ranking, test_users_idx, item_ids)