import numpy as np
import scipy.sparse as sps
from collections import namedtuple
from sklearn.model_selection import KFold, ParameterGrid
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator
import time
import pandas as pd
import sys
sys.path.append('./../')
import utils.utils as ut
from TopPopular.TopPopular import TopPop

def cv_search(rec, urm, icm, non_active_items_mask, sample_size, sample_from_urm=True):
    np.random.seed(1)
    urm_sample, icm_sample, _, non_active_items_mask_sample = ut.produce_sample(urm, icm=icm, ucm=None,
                                                                                 non_active_items_mask=non_active_items_mask,
                                                                                 sample_size=sample_size, sample_from_urm=sample_from_urm)
    params = {'C_SVM': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1], 'k_nn':[20000], 'sh':[2000]}
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
            break
        print maps
        results.append(result(np.mean(maps), np.std(maps), pars))
        print "Result: ", result(np.mean(maps), np.std(maps), pars)
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP", "Std"] + _.parameters.keys())
    print "Total scores: ", scores
    scores.to_csv('Item_SCM CV MAP values 3.csv', sep='\t', index=False)
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



class Item_fSCM(BaseEstimator):
    def __init__(self, top_pops, alpha_ridge=None, C_logreg=None, C_SVM=None, k_nn=2000, sh=2000,pred_batch_size=2500):
        self.alpha_ridge = alpha_ridge
        self.C_logreg = C_logreg
        self.C_SVM = C_SVM
        self.top_pops = top_pops
        self.k_nn = k_nn
        self.sh = sh
        self.pred_batch_size = pred_batch_size

    def fit(self, URM, ICM):
        print time.time(), ": ", "Started fit"
        URM = ut.check_matrix(URM, 'csc', dtype=np.float32)
        n_items = URM.shape[1]

        if self.alpha_ridge is not None:
            self.model = RidgeClassifier(self.alpha_ridge, copy_X=False, fit_intercept=False)
        elif self.C_logreg is not None:
            self.model = LogisticRegression(C=self.C_logreg, solver='newton-cg', fit_intercept=False)
            #TODO: try different values for solver and muti_class
        else:
            self.model = LinearSVC(C=self.C_SVM, fit_intercept=False)

        #values, rows, cols = [[],[],[],[]], [[],[],[],[]], [[],[],[],[]]
        values, rows, cols = [], [], []

        for j in np.sort(self.top_pops):
            top_k_idx = get_knn_CB(ICM, j, self.k_nn, self.sh)
            y = URM[:, j].toarray().ravel()
            y[y > 0.0] = 1.0
            startptr = URM.indptr[j]
            endptr = URM.indptr[j + 1]
            bak = URM.data[startptr: endptr].copy()
            URM.data[startptr: endptr] = 0.0
            try:
                self.model.fit(URM[:,top_k_idx], y)
                coefs = self.model.coef_.ravel()
                nnz_mask = coefs > 0.0

                #for i in range(4):
                #    values[i].extend(self.model.coef_[nnz_mask[i]])
                #    rows[i].extend(np.arange(n_items)[nnz_mask[i]])
                #    cols[i].extend(np.ones(nnz_mask[i].sum()) * j)
                #print nnz_mask.sum(), (self.model.coef_ > 1e-4).sum()

                values.extend(coefs[nnz_mask])
                rows.extend(np.arange(n_items)[nnz_mask])
                cols.extend(np.ones(nnz_mask.sum()) * j)
            except ValueError:
                pass
            URM.data[startptr:endptr] = bak

        #for i in range(4):
        #    self.W_sparse[i] = sps.csc_matrix((values[i], (rows[i], cols[i])), shape=(n_items, n_items), dtype=np.float32)
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        print time.time(), ": ", "Finished fit"

    def predict(self, URM, n_of_recommendations=5, non_active_items_mask=None):
        print time.time(), ": ", "Started predict"
        # compute the scores using the dot product
        user_profile = URM

        n_iterations = user_profile.shape[0] / self.pred_batch_size + (user_profile.shape[0] % self.pred_batch_size != 0)
        ranking = None

        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations
            start = i * self.pred_batch_size
            end = start + self.pred_batch_size if i < n_iterations - 1 else user_profile.shape[0]

            batch_profiles = URM[start:end,:]

            if self.C_SVM is not None:
                batch_scores = batch_profiles.dot(self.W_sparse).toarray().astype(np.float32)
            else:
                batch_scores = -batch_profiles.dot(self.W_sparse).toarray().astype(np.float32)
                np.exp(batch_scores, batch_scores)
                batch_scores += 1
                np.reciprocal(batch_scores, batch_scores)

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

def get_knn_CB(icm, i, k, sh):
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


urm = ut.read_interactions()
items_dataframe = ut.read_items()
item_ids = items_dataframe.id.values
icm = ut.generate_icm(items_dataframe)
icm = ut.normalize_matrix(icm, row_wise=True)
actives = np.array(items_dataframe.active_during_test.values)
non_active_items_mask = actives == 0
test_users_idx = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
urm_pred = urm[test_users_idx, :]

top_rec = TopPop(count=True)
top_rec.fit(urm)
top_pops = top_rec.top_pop[non_active_items_mask[top_rec.top_pop] == False]

# TODO: Use all top_pops or only active ones in fitting??
recommender = Item_fSCM(top_pops=top_pops, pred_batch_size=1000)
#recommender.fit(urm)
cv_search(recommender, urm, icm, non_active_items_mask, sample_size=10000, sample_from_urm=True)

'''urm[urm > 0] = 1
recommender = Item_fSCM(top_pops=top_pops, pred_batch_size=1000)
cv_search(recommender, urm, icm, non_active_items_mask, sample_size=10000, sample_from_urm=True)'''