import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator
import sys
import time
import pandas as pd

sys.path.append('./../')
import utils.utils as ut


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

    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True):
        super(SLIM_recommender, self).__init__()
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)

    def __str__(self):
        return "SLIM (l1_penalty={},l2_penalty={},positive_only={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only
        )

    def fit(self, URM, top_pops=None):
        print time.time(), ": ", "Started fit"
        self.dataset = URM
        URM = ut.check_matrix(URM, 'csc', dtype=np.float32)
        n_items = URM.shape[1]

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False)

        # we'll store the W matrix into a sparse csr_matrix
        # let's initialize the vectors used by the sparse.csc_matrix constructor
        values, rows, cols = [], [], []

        # fit each item's factors sequentially (not in parallel)
        for j in (range(n_items) if top_pops is None else top_pops):
            print time.time(), ": ", "Started fit > Iteration ", j, "/", n_items
            # get the target column
            y = URM[:, j].toarray()
            # set the j-th column of X to zero
            startptr = URM.indptr[j]
            endptr = URM.indptr[j + 1]
            bak = URM.data[startptr: endptr].copy()
            URM.data[startptr: endptr] = 0.0
            # fit one ElasticNet model per column
            print time.time(), ": ", "Started fit > Iteration ", j, "/", n_items, " > Fitting ElasticNet model"
            self.model.fit(URM, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            nnz_mask = self.model.coef_ > 0.0
            values.extend(self.model.coef_[nnz_mask])
            rows.extend(np.arange(n_items)[nnz_mask])
            cols.extend(np.ones(nnz_mask.sum()) * j)

            # finally, replace the original values of the j-th column
            URM.data[startptr:endptr] = bak

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def predict(self, URM, n_of_recommendations=5, non_active_items_mask=None):
        print time.time(), ": ", "Started predict"
        # compute the scores using the dot product
        user_profile = URM
        print time.time(), ": ", "Started predict > Started dot product"
        scores = user_profile.dot(self.W_sparse).toarray()
        print np.extract(scores != 0, scores).shape
        # exclude seen and non active
        non_zero_indices = user_profile.nonzero()
        scores[non_zero_indices[0], non_zero_indices[1]] = 0.0
        scores[:, non_active_items_mask] = 0.0

        # rank items
        print time.time(), ": Started predict > Started argsort"
        ranking = scores.argsort()[:, ::-1]
        ranking = ranking[:, :n_of_recommendations]

        return ranking


urm = ut.read_interactions()
recommender = SLIM_recommender(l1_penalty=0.000001, l2_penalty=1000)
top_pops = ut.read_top_pops()
recommender.fit(urm, top_pops=top_pops)

items_dataframe = ut.read_items()
actives = np.array(items_dataframe.active_during_test.values)
non_active_items_mask = actives == 0

test_users_idx = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
urm = urm[test_users_idx, :]
item_ids = items_dataframe.id.values
ranking = recommender.predict(urm, non_active_items_mask=non_active_items_mask)

ut.write_recommendations("SLIM_shady", ranking, test_users_idx, item_ids)
