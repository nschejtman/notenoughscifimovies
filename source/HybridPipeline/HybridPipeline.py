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

def save_sparse_matrix(mat, name):
    np.save(name + ' data', mat.data)
    np.save(name + ' indptr', mat.indptr)
    np.save(name + ' indices', mat.indices)
    np.save(name + ' shape', mat.shape)


def load_sparse_matrix(name, format, type):
    if format == 'csr':
        mat = sps.csr_matrix((np.load(name + ' data.npy'),
                              np.load(name + ' indices.npy'),
                              np.load(name + ' indptr.npy')),
                             np.load(name + ' shape.npy'),
                             dtype=type)
    elif format == 'csc':
        mat = sps.csc_matrix((np.load(name + ' data.npy'),
                              np.load(name + ' indices.npy'),
                              np.load(name + ' indptr.npy')),
                             np.load(name + ' shape.npy'),
                             dtype=type)
    return mat

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

    # TODO: Use all top_pops or only active ones in fitting??
    fslim = IfSL.fSLIM_recommender(top_pops=top_pops, pred_batch_size=1000, sim_partition_size=1000, k_nn=30000,
                                    similarity='CB', aa_sh=2000, alpha_ridge=100000)
    # fslim.fit(urm_implicit, icm)
    # save_sparse_matrix(fslim.W_sparse, "fSLIM 30000knn 100000alpha 2000sh CBsim ratings1")
    fslim.W_sparse = load_sparse_matrix("fSLIM 30000knn 100000alpha 2000sh CBsim ratings1", 'csc', np.float32)
    '''ranking_fslim = fslim.predict(urm_explicit, 5, non_active_items_mask)
    ranking_fslim = np.array(ranking_fslim).flatten()
    np.save("ranking5 fslim 1", ranking_fslim)'''
    ranking_fslim = np.load("ranking5 fslim.npy")


    '''data = np.ones(ranking_fslim.shape[0])
    cols = ranking_fslim.copy()
    rows = np.repeat(range(urm_implicit.shape[0]),5)
    urm_enriched = sps.csr_matrix((data, (rows,cols)), urm_implicit.shape)
    urm_enriched[urm_implicit != 0] = 1
    save_sparse_matrix(urm_enriched, "enriched 1")'''
    urm_enriched = load_sparse_matrix("enriched",'csr', np.int64)

    scm = ISC.Item_SCM(top_pops=top_pops, pred_batch_size=1000, C_SVM=0.01)
    # scm.fit(urm_enriched)
    # save_sparse_matrix(scm.W_sparse, "SCM SVM enriched 1minus2C ratings1")
    scm.W_sparse = load_sparse_matrix("SCM SVM enriched 1minus2C ratings1", 'csc', np.float32)
    ranking_scm = scm.predict(urm_implicit[test_users_idx,:], 5, non_active_items_mask)
    ut.write_recommendations("HybridPipeline fSLIM 30000k 100000alpha 2000sh implrating CBsim SCM SVM 1minus2 impllpred", ranking_scm, test_users_idx, item_ids)

    # fslim.fit(urm_enriched, icm)
    # save_sparse_matrix(fslim.W_sparse, "fSLIM 30000knn 100000alpha 2000sh CBsim ratings1 enriched")
    fslim.W_sparse = load_sparse_matrix("fSLIM 30000knn 100000alpha 2000sh CBsim ratings1 enriched", 'csc', np.float64)
    ranking_fslim_enriched = fslim.predict(urm_pred, 5, non_active_items_mask)
    ut.write_recommendations("HybridPipeline fSLIM 30000k 100000alpha 2000sh implrating CBsim x2", ranking_fslim_enriched, test_users_idx, item_ids)

main()