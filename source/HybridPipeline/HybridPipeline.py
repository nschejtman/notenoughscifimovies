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
from Item_fSLIM import ItemfSLIM as IfSL
from Item_SCM import Item_SCM as ISC


def cv_search(rec, urm_expl, urm_impl, icm, non_active_items_mask, sample_size, filename):
    np.random.seed(1)
    perm = np.random.permutation(urm_expl.shape[0])[:sample_size]
    icm_sample = icm
    non_active_items_mask_sample = non_active_items_mask
    urm_sample = urm_impl[perm]
    params = {'alpha_ridge':[9500, 9750, 10000, 25000, 50000, 75000, 100000], 'k_nn':[30000],
              'similarity':['CB'], 'aa_sh':[2000]}
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
            urm_test = urm_expl[perm][row_test,:]
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
    scores.to_csv(filename+'.csv', sep='\t', index=False)


def produce_enriched_urm(rec, icm, urm_original_fit, urm_original_pred, n_predictions, non_active_items_mask, load_matrix, filename):
    if load_matrix:
        rec.W_sparse = ut.load_sparse_matrix(filename, 'csc', np.float32)
    else:
        rec.fit(urm_original_fit, icm)
        ut.save_sparse_matrix(rec.W_sparse, filename)

    ranks = rec.predict(urm_original_pred, n_predictions, non_active_items_mask)
    ranks = np.array(ranks).flatten()

    data = np.ones(ranks.shape[0])
    cols = ranks.copy()
    rows = np.repeat(range(urm_original_fit.shape[0]),n_predictions)
    urm_enriched = sps.csr_matrix((data, (rows,cols)), urm_original_fit.shape)
    urm_enriched[urm_original_fit != 0] = 1
    return urm_enriched



def main():
    urm_explicit = ut.read_interactions()
    urm_implicit = urm_explicit.copy()
    urm_implicit[urm_explicit > 0] = 1
    items_dataframe = ut.read_items()
    icm = ut.generate_icm(items_dataframe)
    icm = ut.normalize_matrix(icm, row_wise=True)
    item_ids = items_dataframe.id.values
    actives = np.array(items_dataframe.active_during_test.values)
    non_active_items_mask = actives == 0
    test_users_idx = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
    urm_pred = urm_explicit[test_users_idx, :]

    top_rec = TopPop(count=True)
    top_rec.fit(urm_implicit)
    top_pops = top_rec.top_pop[non_active_items_mask[top_rec.top_pop] == False]

    fslim = IfSL.fSLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                    similarity='CB', aa_sh=2000, alpha_ridge=100000)
    urm_enriched = produce_enriched_urm(fslim, icm, urm_implicit, urm_explicit, 2, non_active_items_mask, True, "fSLIM 30000knn 100000alpha 2000sh CBsim ratings1")

    # ----------------------------- SECOND PHASE ---------------------------------
    fslim = IfSL.fSLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                    similarity='CB', aa_sh=2000)
    #cv_search(fslim, urm_expl=urm_explicit, urm_impl=urm_enriched, icm=icm, non_active_items_mask=non_active_items_mask,
    #          sample_size=10000, filename='fSLIM (Ridge) CV MAP values enriched phase 2')
    # TODO: select best alpha
    fslim = IfSL.fSLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                   similarity='CB', aa_sh=2000, alpha_ridge=100000)
    urm_enriched = produce_enriched_urm(fslim, icm, urm_enriched, urm_explicit, 2, non_active_items_mask, False,
                                        "fSLIM 30000knn 100000alpha 2000sh CBsim ratings1+2 enriched phase 2")

    # ----------------------------- THIRD PHASE ---------------------------------
    fslim = IfSL.fSLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                   similarity='CB', aa_sh=2000)
    #cv_search(fslim, urm_expl=urm_explicit, urm_impl=urm_enriched, icm=icm, non_active_items_mask=non_active_items_mask,
    #          sample_size=10000, filename='fSLIM (Ridge) CV MAP values enriched phase 3')
    # TODO: select best alpha
    fslim = IfSL.fSLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                   similarity='CB', aa_sh=2000, alpha_ridge=100000)
    urm_enriched = produce_enriched_urm(fslim, icm, urm_enriched, urm_explicit, 1, non_active_items_mask, False,
                                        "fSLIM 30000knn 100000alpha 2000sh CBsim ratings1+4  enriched phase 3")

    # ----------------------------- FOURTH PHASE ---------------------------------
    fslim = IfSL.fSLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                   similarity='CB', aa_sh=2000)
    # cv_search(fslim, urm_expl=urm_explicit, urm_impl=urm_enriched, icm=icm, non_active_items_mask=non_active_items_mask,
    #         sample_size=10000, filename='fSLIM (Ridge) CV MAP values enriched phase 4 (final)')
    # TODO: select best alpha
    fslim = IfSL.fSLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                   similarity='CB', aa_sh=2000, alpha_ridge=100000)
    fslim.fit(urm_enriched, icm)
    ut.save_sparse_matrix(fslim.W_sparse, "fSLIM 30000knn 100000alpha 2000sh CBsim ratings1+5 enriched phase 4")
    ranks = fslim.predict(urm_pred, 5, non_active_items_mask)
    ut.write_recommendations("HybridPipeline fSLIM 30000k 100000alpha 2000sh implrating CBsim x4", ranks, test_users_idx, item_ids)


    '''scm = ISC.Item_SCM(top_pops=top_pops, pred_batch_size=1000, C_SVM=0.01)
    # scm.fit(urm_enriched)
    # save_sparse_matrix(scm.W_sparse, "SCM SVM enriched 1minus2C ratings1")
    scm.W_sparse = load_sparse_matrix("SCM SVM enriched 1minus2C ratings1", 'csc', np.float32)
    ranking_scm = scm.predict(urm_implicit[test_users_idx,:], 5, non_active_items_mask)
    ut.write_recommendations("HybridPipeline fSLIM 30000k 100000alpha 2000sh implrating CBsim SCM SVM 1minus2 impllpred", ranking_scm, test_users_idx, item_ids)

    # fslim.fit(urm_enriched, icm)
    # save_sparse_matrix(fslim.W_sparse, "fSLIM 30000knn 100000alpha 2000sh CBsim ratings1 enriched")
    fslim.W_sparse = load_sparse_matrix("fSLIM 30000knn 100000alpha 2000sh CBsim ratings1 enriched", 'csc', np.float64)
    ranking_fslim_enriched = fslim.predict(urm_pred, 5, non_active_items_mask)
    ut.write_recommendations("HybridPipeline fSLIM 30000k 100000alpha 2000sh implrating CBsim x2", ranking_fslim_enriched, test_users_idx, item_ids)'''

main()