import numpy as np
import scipy.sparse as sps
import time
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import utils.utils as ut
import matplotlib as mpl
from collections import namedtuple
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.base import BaseEstimator
from TopPopular.TopPopular import TopPop
from multiprocessing import Pool
from functools import partial

mpl.use('Agg')
sys.path.append('./../')


# noinspection SpellCheckingInspection
def cv_search(rec, urm, icm, non_active_items_mask, sample_size, sample_from_urm=True):
    np.random.seed(1)
    urm_sample, icm_sample, _, non_active_items_mask_sample = ut.produce_sample(urm,
                                                                                icm=icm,
                                                                                ucm=None,
                                                                                non_active_items_mask=
                                                                                non_active_items_mask,
                                                                                sample_size=sample_size,
                                                                                sample_from_urm=sample_from_urm)
    params = {'alpha_ridge': [9500, 9750, 10000, 25000, 50000, 75000, 100000], 'k_nn': [20000],
              'similarity': ['CF'], 'aa_sh': [500, 1000, 2000]}
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

        for row_train, row_test in splits:
            urm_train = urm_sample[row_train, :]
            rec.fit(urm_train, icm_sample)
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
            maps.append(ut.map_scorer(rec, urm_test, hidden_ratings, n,
                                      non_active_items_mask_sample))  # Assume rec to predict indices of items, NOT ids
            print "Progress: {:.2f}%".format((prog * 100) / total)
            prog += 1
            break
        print maps
        results.append(result(np.mean(maps), np.std(maps), pars))
        print "Result: ", result(np.mean(maps), np.std(maps), pars)
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP", "Std"] + _.parameters.keys())
    print "Total scores: ", scores
    scores.to_csv('fSLIM_Item CV MAP values 3.csv', sep='\t', index=False)
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


# noinspection PyPep8Naming,PyAttributeOutsideInit
class fSLIM_recommender(BaseEstimator):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self, train_items, l1_ratio=None, positive_only=True, alpha_ridge=None, alpha_lasso=None,
                 similarity='CF', k_nn=10, aa_sh=2000, sim_partition_size=2500, pred_batch_size=2500):
        super(fSLIM_recommender, self).__init__()
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.alpha_ridge = alpha_ridge
        self.alpha_lasso = alpha_lasso
        self.k_nn = k_nn
        self.similarity = similarity
        self.sh = aa_sh
        self.sim_partition_size = sim_partition_size
        self.train_items = train_items
        self.pred_batch_size = pred_batch_size

    def fit(self, URM, ICM):
        print time.time(), ": ", "Started fit"

        URM = ut.check_matrix(URM, 'csc', dtype=np.float32)  # Check matrix type
        n_items = URM.shape[1]
        values, rows, cols = [], [], []

        # Select the right model to instantiate
        if self.alpha_ridge is not None:

            self.model = Ridge(self.alpha_ridge,
                               copy_X=False,
                               fit_intercept=False)

        elif self.alpha_lasso is not None:

            self.model = Lasso(alpha=self.alpha_lasso,
                               copy_X=False,
                               fit_intercept=False)
        else:
            """
            Minimize
            1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
            """
            self.model = ElasticNet(alpha=1.0,
                                    l1_ratio=self.l1_ratio,
                                    positive=self.positive_only,
                                    fit_intercept=False,
                                    copy_X=False)

        for j in np.sort(self.train_items):

            # Get nearest KNN
            if self.similarity == 'CB':
                knn_indices = get_item_knn_CB(ICM, j, self.k_nn, self.sh)
            else:
                knn_indices = get_item_knn_CF(URM, j, self.k_nn, self.sh)

            y = URM[:, j].toarray()  # Ratings for item j

            # Fit the right model
            if self.alpha_ridge is None and self.alpha_lasso is None:
                self.model.fit(URM[:, knn_indices], y)
            else:
                self.model.fit(URM[:, knn_indices], y.ravel())

            # Construct the 3 arrays that compose a CSC matrix separately
            nnz_mask = self.model.coef_ > 0.0
            values.extend(self.model.coef_[nnz_mask])
            rows.extend(knn_indices[nnz_mask])
            cols.extend(np.ones(nnz_mask.sum()) * j)

        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        print time.time(), ": ", "Finished fit"

    def predict(self, URM, test_users_idx, UCM, n_of_recommendations=5, non_active_items_mask=None):

        print time.time(), ": ", "Started predict"

        user_profile = URM[test_users_idx]

        n_iterations = user_profile.shape[0] / \
                       self.pred_batch_size + (user_profile.shape[0] % self.pred_batch_size != 0)

        ranking = None

        # Predict by batches
        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations

            # Calculate batch indices
            start = i * self.pred_batch_size
            end = start + self.pred_batch_size if i < n_iterations - 1 else user_profile.shape[0]

            batch_profiles = user_profile[start:end, :]
            batch_scores = batch_profiles.dot(self.W_sparse).toarray().astype(np.float32)

            nonzero_indices = batch_profiles.nonzero()
            batch_scores[nonzero_indices[0], nonzero_indices[1]] = 0.0

            # Remove inactive items
            batch_scores[:, non_active_items_mask] = 0.0
            batch_ranking = batch_scores.argsort()[:, ::-1]
            batch_ranking = batch_ranking[:, :n_of_recommendations]  # Leave only the top n

            if i == 0:
                ranking = batch_ranking.copy()
            else:
                ranking = np.vstack((ranking, batch_ranking))

        """
            Set ranking for empty profile users

        """
        # Get mask for URM of users with no profile
        sum_of_users_ratings = np.array(URM.sum(axis=1)).flatten()
        empty_profile_users_mask = sum_of_users_ratings == 0

        # Get mask for URM of users in test set
        test_users_mask = np.zeros(URM.shape[0], dtype=bool)
        test_users_mask[test_users_idx] = True

        # Get mask for users in test set AND with empty profile
        empty_profile_test_users_mask = np.logical_and(test_users_mask, empty_profile_users_mask)

        # Get KNNs for users in test set and with empty profile
        knn_empty_profile_users = get_user_knn_CB(UCM, empty_profile_test_users_mask, empty_profile_users_mask, 1, self.sh).ravel()

        # Predict for KNNs
        knn_profiles = URM[knn_empty_profile_users,:]
        knn_scores = knn_profiles.dot(self.W_sparse).toarray().astype(np.float32)

        # Remove inactive items for predictions
        knn_scores[:, non_active_items_mask] = 0.0
        knn_ranking = knn_scores.argsort()[:, ::-1]
        knn_ranking = knn_ranking[:, :n_of_recommendations]  # Leave only the top n rec

        # Add predictions to ranking (N.B. apply mask for ranking rather than URM for test users with empty profile)
        ranking[empty_profile_test_users_mask[test_users_idx]] = knn_ranking

        print time.time(), ": ", "Finished predict"
        return ranking


def get_item_knn_CB(icm, i, k, sh):
    # icm = ut.normalize_matrix(icm, row_wise=True)
    sims = icm[i, :].dot(icm.T).toarray().ravel()
    sims[i] = 0.0
    icm_ind = icm.copy()
    icm_ind.data = np.ones_like(icm_ind.data)
    counts = icm_ind[i, :].dot(icm_ind.T).toarray().ravel()
    counts /= (counts + sh)
    sims *= counts
    top_k = np.argsort(sims).ravel()
    return top_k[-k:]


def get_item_knn_CF(urm, i, k, sh):
    urm_copy = urm.copy()
    urm_copy = ut.normalize_matrix(urm_copy, row_wise=False)
    sims = urm_copy[:, i].T.dot(urm_copy).toarray().ravel()
    sims[i] = 0.0
    urm_ind = urm_copy.copy()
    urm_ind.data = np.ones_like(urm_ind.data)
    counts = urm_ind[:, i].T.dot(urm_ind).toarray().ravel()
    counts /= (counts + sh)
    sims *= counts
    top_k = np.argsort(sims).ravel()
    return top_k[-k:]


def get_user_knn_CB(ucm, test_users_mask, empty_profile_users_mask, k, sh):
    """
    Takes as input the UCM (user content matrix) and returns an array of k user indices corresponding to the KNN of a
    particular user taking into consideration it's user features rather than it's ratings.

    :param ucm: user content matrix
    :param test_users_mask: users to calculate similarity
    :param empty_profile_users_mask: users with empty profile
    :param k: desired number of nearest neighbours
    :param sh: shrink

    :return: indices in ucm KNN for all users in users mask
    """
    ucm_copy = ucm.copy()
    sims = ucm_copy[test_users_mask, :].dot(ucm.T).toarray()

    # Set similarity of each user in the mask with itself and all the other users in the mask to 0
    # sims[np.arange(sims.shape[0]), users_mask.T] = 0.0
    for u in range(sims.shape[0]):
        sims[u, empty_profile_users_mask] = 0.0

    # Apply shrink
    ucm_ind = ucm_copy.copy()
    ucm_ind.data = np.ones_like(ucm_ind.data)

    counts = ucm_ind[test_users_mask, :].dot(ucm_ind.T).toarray()
    counts /= (counts + sh)
    sims *= counts

    top_k = np.argsort(sims)

    return top_k[:,-k:]


def calculate_and_save_similarities(max_k, shs, partition_size):
    items_dataframe = ut.read_items()
    icm = ut.generate_icm(items_dataframe)
    for sh in shs:
        sim, top_k_idx = ut.compute_similarity_matrix_knn(icm, max_k, sh, row_wise=True, partition_size=partition_size)
        np.save('Similarity' + str(max_k) + '_' + str(sh) + 'data', sim.data)
        np.save('Similarity' + str(max_k) + '_' + str(sh) + 'indptr', sim.indptr)
        np.save('Similarity' + str(max_k) + '_' + str(sh) + 'indices', sim.indices)
        np.save('Similarity' + str(max_k) + '_' + str(sh) + 'shape', sim.shape)
        np.save('Similarity' + str(max_k) + '_' + str(sh) + 'top_k_idx', top_k_idx)


def load_similarities(max_k, sh):
    sim = sps.csr_matrix((np.load('Similarity' + str(max_k) + '_' + str(sh) + 'data.npy'),
                          np.load('Similarity' + str(max_k) + '_' + str(sh) + 'indices.npy'),
                          np.load('Similarity' + str(max_k) + '_' + str(sh) + 'indptr.npy')),
                         np.load('Similarity' + str(max_k) + '_' + str(sh) + 'shape.npy'))
    top_k_idx = np.load('Similarity' + str(max_k) + '_' + str(sh) + 'top_k_idx.npy')
    return sim, top_k_idx


def main():
    urm = ut.read_interactions()

    items_dataframe = ut.read_items()
    # icm = ut.generate_icm(items_dataframe)
    # icm = ut.generate_icm_unified_attrs(items_dataframe)
    ucm = ut.generate_ucm()
    # icm = ut.normalize_matrix(icm, row_wise=True)
    ucm = ut.normalize_matrix(ucm, row_wise=True)
    item_ids = items_dataframe.id.values
    actives = np.array(items_dataframe.active_during_test.values)
    non_active_items_mask = actives == 0
    test_users_idx = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
    urm_pred = urm[test_users_idx, :]

    tp_recommender = TopPop(count=True)
    tp_recommender.fit(urm)
    train_items = tp_recommender.top_pop[non_active_items_mask[tp_recommender.top_pop] == False]

    urm[urm > 0] = 1
    # print urm.data[:10]
    # urm_aux = ut.urm_to_tfidf(urm)
    # print urm_aux.data[:10]

    recommender = fSLIM_recommender(train_items=train_items, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                    similarity='CB', aa_sh=2000, alpha_ridge=100000)
    # recommender.fit(urm, icm)
    recommender.W_sparse = ut.load_sparse_matrix(
        '../Hybrid_SLIM_Sim_fSLIM_pred/fSLIM 30000knn 100000alpha 2000sh CBsim ratings1 phase 2', 'csc', np.float32)
    ranking = recommender.predict(urm, test_users_idx, ucm, 5, non_active_items_mask)
    ut.write_recommendations("asd", ranking,
                             test_users_idx, item_ids)


main()
