import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.base import BaseEstimator
import sys
sys.path.append('./../')
import utils.utils as ut


def cv_search(rec, urm, non_active_items_mask, sample_size, sample_from_urm=True):
    np.random.seed(1)
    urm_sample, _, _, non_active_items_mask_sample = ut.produce_sample(urm, icm=None, ucm=None,
                                                                                 non_active_items_mask=non_active_items_mask,
                                                                                 sample_size=sample_size, sample_from_urm=sample_from_urm)
    params = {'count': [True, False]}
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
        rec = rec.set_params(**pars)
        maps = []

        for row_train, row_test in splits:
            urm_train = urm_sample[row_train,:]
            rec.fit(urm_train)
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
    scores.to_csv('TopPop CV MAP values.csv', sep='\t', index=False)


class TopPop(BaseEstimator):

    def __init__(self, count=False):
        self.top_pop = None
        self.count = count

    def fit(self, urm):
        if self.count:
            urm_ind = urm.copy()
            urm_ind.data = np.ones_like(urm_ind.data)
            self.top_pop = urm_ind.sum(axis=0)
        else:
            self.top_pop = urm.sum(axis=0)
        self.top_pop = np.asarray(self.top_pop).squeeze()
        self.top_pop = np.argsort(self.top_pop)[::-1]

    def predict(self, urm, n, non_active_items_mask):
        ranking = []
        for user in range(urm.shape[0]):
            rated_items = (urm[user,:] != 0).toarray().ravel()
            user_ratings = self.top_pop[non_active_items_mask[self.top_pop] == False &
                                        (rated_items[self.top_pop] == False)][:n]
            ranking.append(user_ratings)
        return ranking


'''urm = ut.read_interactions()
items_dataframe = ut.read_items()
actives = np.array(items_dataframe.active_during_test.values)
non_active_items_mask = actives == 0
recommender = TopPop()
recommender.fit(urm)
cv_search(recommender, urm, non_active_items_mask, sample_size=10000, sample_from_urm=True)
top_pops = recommender.top_pop[non_active_items_mask[recommender.top_pop] == False]
pd.DataFrame(top_pops).to_csv('../../inputs/top_pop_sum_idx.csv', sep='\t', index=False)'''