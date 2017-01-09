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
    params = {'sh': [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500][::-1]}
    params = {'sh': [100, 200, 500, 1000, 2000, 5000, 10000, 20000]}
    grid = list(ParameterGrid(params))
    folds = 2
    kfold = KFold(n_splits=folds)
    splits = [(train, test) for train,test in kfold.split(urm_sample)]
    hidden_ratings_perc = 0.75
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
            rec.fit(icm_sample)
            urm_test = urm_sample[row_test,]
            hidden_ratings = []
            for u in range(urm_test.shape[0]):
                relevant_u = urm_test[u,].nonzero()[1]  # Indices of rated items for test user u
                if len(relevant_u) > 1:#1 or 2
                    np.random.shuffle(relevant_u)
                    urm_test[u, relevant_u[int(len(relevant_u) * hidden_ratings_perc):]] = 0
                    hidden_ratings.append(relevant_u[int(len(relevant_u) * hidden_ratings_perc):])
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
    scores.to_csv('ItemCB_BF CV MAP values 7.csv', sep='\t', index=False)
    '''f = scores['MAP'].plot(title="ItemCB_BF CV MAP values").get_figure()
    x_max, y_max = scores['MAP'].argmax(), scores['MAP'].max()
    plt.plot(x_max, y_max, 'o', color='r')
    plt.figtext(0, 0,"Not normalized ratings, No title/tags\nMaximumm at (sh={:.5f},{:.5f}+/-{:.5f})".format(x_max,
                                                                                                          y_max,
                                                                                                          scores['Std'][x_max]))
    f.savefig('ItemCB_BF CV MAP values 2.png', bbox_inches='tight')
    plt.close(f)'''




class ItemCB_BF(BaseEstimator):
    def __init__(self, top_pops, sh=2, pred_batch_size=1000, normalize=False):
        self.sh = sh
        self.pred_batch_size = pred_batch_size
        self.icm = None
        self.normalize = normalize
        self.top_pops = top_pops

    def fit(self, icm):
        self.icm = icm

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
            batch_sim_mat = ut.compute_similarity_matrix_mask(self.icm, self.sh, rated_items_batch)
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
                batch_ranking[zero_scores_mask] = [self.top_pops[:n] for _ in range(n_zero_scores)]

            if i == 0:
                ranking = batch_ranking.copy()
            else:
                ranking = np.vstack((ranking, batch_ranking))

        return ranking


def main():
    # Read items
    items_dataframe = ut.read_items()
    urm = ut.read_interactions()
    actives = np.array(items_dataframe.active_during_test.values)
    non_active_items_mask = actives == 0
    item_ids = items_dataframe.id.values
    icm = ut.generate_icm(items_dataframe)

    top_rec = TopPop(count=True)
    top_rec.fit(urm)
    top_pops = top_rec.top_pop[non_active_items_mask[top_rec.top_pop] == False]
    recommender = ItemCB_BF(top_pops=top_pops,sh=0.1, pred_batch_size=200, normalize=False)
    #cv_search(recommender, urm, icm, non_active_items_mask, sample_size=10000, sample_from_urm=True)

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

main()