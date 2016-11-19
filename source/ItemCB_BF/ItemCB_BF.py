from collections import namedtuple
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, ParameterGrid
import sys
sys.path.append('./../')
import utils.utils as ut
from TopPopular.TopPopular import TopPop


'''
Perform CV search on parameters. If sample_from_urm is True then sample_sie specifices number of rows from the urm to be kept;
otherwise it specifices number of interactions to be kept. When sample_size is None, we just get a randomization of data
'''
def cv_search(rec, urm, icm, non_active_items_mask, sample_size, sample_from_urm=True):
    np.random.seed(1)
    urm_sample, icm_sample, _, non_active_items_mask_sample = ut.produce_sample(urm, icm=icm, ucm=None,
                                                                                 non_active_items_mask=non_active_items_mask,
                                                                                 sample_size=sample_size, sample_from_urm=sample_from_urm)
    urm_sample = urm
    params = {'sh': [0.1, 0.5, 1, 2, 5, 10, 20, 50]}
    grid = list(ParameterGrid(params))
    folds = 2
    kfold = KFold(n_splits=folds)
    splits = [(train, test) for train,test in kfold.split(urm_sample)]
    retained_ratings_perc = 0.75
    n = 5
    result = namedtuple('result', ['mean_score', 'std_dev', 'parameters'])
    results = []
    total = float(reduce(lambda acc, x: acc * len(x), params.itervalues(), 1) * folds)
    prog = 1.0

    for pars in grid:
        rec = rec.set_params(**pars)
        maps = []

        for row_train, row_test in splits:
            urm_train = urm_sample[row_train,]
            rec.fit(icm_sample, urm_sample)
            urm_test = urm_sample[row_test,]
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
        results.append(result(np.mean(maps), np.std(maps), pars))
        print "Result: ", result(np.mean(maps), np.std(maps), pars)
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP", "Std"] + _.parameters.keys())
    print "Total scores: ", scores
    scores.to_csv('ItemCB_BF CV MAP values 2.csv', sep='\t', index=False)
    f = scores['MAP'].plot(title="ItemCB_BF CV MAP values").get_figure()
    x_max, y_max = scores['MAP'].argmax(), scores['MAP'].max()
    plt.plot(x_max, y_max, 'o', color='r')
    plt.figtext(0, 0,"Not normalized ratings, No title/tags\nMaximumm at (sh={:.5f},{:.5f}+/-{:.5f})".format(x_max,
                                                                                                          y_max,
                                                                                                          scores['Std'][x_max]))
    f.savefig('ItemCB_BF CV MAP values 2.png', bbox_inches='tight')
    plt.close(f)


def holdout_search(rec, urm, icm, actives, sample_size=None):
    np.random.seed(1)
    nnz = urm.nonzero()
    perm = np.random.permutation(len(nnz[0]))[:sample_size]
    unique_rows = np.unique(nnz[0][perm])
    nnz_to_sample_row = {unique_rows[i]: i for i in range(len(unique_rows))}
    URM_sample = sps.lil_matrix((unique_rows.size, urm.shape[1]))
    for i in perm:
        URM_sample[nnz_to_sample_row[nnz[0][i]], nnz[1][i]] = urm[nnz[0][i], nnz[1][i]]
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
            rec.fit(URM_train, icm, actives)
            hidden_ratings, test_rows = [], []
            for u in range(URM_test.shape[0]):
                relevant_u = URM_test[u,].nonzero()[1]  # Indices of rated items for test user u
                if len(relevant_u) > 0:
                    hidden_ratings.append(relevant_u)
                    test_rows.append(u)
            maps.append(ut.map_scorer(rec, URM_train[test_rows,], hidden_ratings,
                                   n))  # Assume rec to predict indices of items, NOT ids
            print "Progress: ", (prog * 100) / total, "%"
            prog += 1
        results.append(Result(np.mean(maps), np.std(maps), pars))
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP", "Std"] + _.parameters.keys())
    scores.to_csv('ItmCB_BF CV MAP values 1.csv', sep='\t', index=False)
    cols, col_feat, x_feat = 3, 'sh', 'k'
    f = sns.FacetGrid(data=scores, col=col_feat, col_wrap=cols, sharex=False, sharey=False)
    f.map(plt.plot, x_feat, 'MAP')
    f.fig.suptitle("ItemCB Holdout MAP values")
    i_max, y_max = scores['MAP'].argmax(), scores['MAP5'].max()
    i_feat_max = params[col_feat].index(scores[col_feat][i_max])
    f_max = f.axes[i_feat_max]
    f_max.plot(scores[x_feat][i_max], y_max, 'o', color='r')
    plt.figtext(0, 0, "Not normalized ratings, no title/tags\nmaximum at (sh={:.5f},k={:.5f}, {:.5f}+/-{:.5f})".format(scores[col_feat][i_max],
                                                                                         scores[x_feat][i_max],
                                                                                         y_max,
                                                                                         scores['Std'][i_max]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    f.savefig('ItemCB Holdout MAP values.png', bbox_inches='tight')


class ItemCB_BF(BaseEstimator):
    def __init__(self, sh=2, pred_batch_size=1000, normalize=False):
        self.sh = sh
        self.pred_batch_size = pred_batch_size
        self.icm = None
        self.normalize = normalize
        self.top_pop = TopPop()

    def fit(self, icm, urm):
        self.icm = icm
        # self.top_pop.fit(urm)

    def predict(self, urm, n, non_active_items_mask):
        print "Started prediction"
        user_profile = urm
        n_iterations = user_profile.shape[0] / self.pred_batch_size + (user_profile.shape[0] % self.pred_batch_size != 0)
        ranking = None

        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations

            start = i * self.pred_batch_size
            end = start + self.pred_batch_size if i < n_iterations - 1 else user_profile.shape[0]

            batch_profiles = user_profile[start:end, :]
            rated_items_batch = np.diff(batch_profiles.tocsc().indptr) != 0
            #print "Similarity batch size: ", np.extract(rated_items_batch == True, rated_items_batch).shape[0]
            #break
            batch_sim_mat = ut.compute_similarity_matrix_batch(self.icm, self.sh, rated_items_batch)
            batch_scores = batch_profiles.dot(batch_sim_mat).toarray().astype(np.float32)

            # normalization
            if self.normalize:
                rated_ind = batch_profiles.copy()
                rated_ind.data = np.ones_like(batch_profiles.data)
                den = rated_ind.dot(batch_sim_mat).toarray().astype(np.float32)
                den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
                batch_scores /= den

            del batch_sim_mat

            # remove the ones that are already rated
            nonzero_indices = batch_profiles.nonzero()
            batch_scores[nonzero_indices[0], nonzero_indices[1]] = 0.0

            # remove the inactives items
            batch_scores[:, non_active_items_mask] = 0.0
            batch_ranking = batch_scores.argsort()[:,::-1]
            batch_ranking = batch_ranking[:,:n]  # leave only the top n

            sum_of_scores = batch_scores[np.arange(batch_scores.shape[0]), batch_ranking.T].T.sum(axis=1).ravel()
            zero_scores_mask = sum_of_scores == 0
            n_zero_scores = np.extract(zero_scores_mask, sum_of_scores).shape[0]
            if n_zero_scores != 0:
                batch_ranking[zero_scores_mask] = [self.top_pop.top_pop[non_active_items_mask[self.top_pop.top_pop] == False][:n] for _ in range(n_zero_scores)]

            if i == 0:
                ranking = batch_ranking.copy()
            else:
                ranking = np.vstack((ranking, batch_ranking))

        return ranking



# Read items

items_dataframe = ut.read_items()
urm = ut.read_interactions()
actives = np.array(items_dataframe.active_during_test.values)
non_active_items_mask = actives == 0
item_ids = items_dataframe.id.values
icm = ut.generate_icm(items_dataframe, include_tags=False, include_title=False)
recommender = ItemCB_BF(sh=0.1, pred_batch_size=200, normalize=False)
top = TopPop(count=True)
top.fit(urm)
recommender.top_pop = top
cv_search(recommender, urm, icm, non_active_items_mask, sample_size=10000, sample_from_urm=True)

'''
test_users_idx = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
urm = urm[test_users_idx, :]
recommender.fit(icm)
recs = recommender.predict(urm, 5, non_active_items_mask)
ut.write_recommendations("Item CB_BF", recs, test_users_idx, item_ids)
'''
'''
urm = sps.csr_matrix([[3,0,2,3,0,0,0,0,0],[0,0,0,0,0,0,2,0,3],[0,0,0,0,1,0,0,0,0],[0,0,0,3,2,1,0,0,0],[0,0,0,0,0,0,0,0,0]])
icm = sps.csr_matrix([[0,4],[1,4],[2,4],[3,4],[4,4],[4,3],[4,2],[4,1],[4,0]])
non_active_items_mask = np.array([False, False, False, False, False, False, False, False, False])
recommender = ItemCB_BF(sh=0, pred_batch_size=2, normalize=False)
'''