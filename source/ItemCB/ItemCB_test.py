from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, ParameterGrid


def map_scorer(rec, urm_test, hidden_ratings, n, ignore_mask, urm_partition_size=1000):
    score = 0
    #rec_list = rec.predict(urm_test, n, ignore_mask, urm_partition_size)
    rec_list = rec.predict2(urm_test, n, ignore_mask, urm_partition_size)
    i = 0
    for u in range(urm_test.shape[0]):
        if len(hidden_ratings[u]) > 0:
            is_relevant = np.in1d(rec_list[u], hidden_ratings[u], assume_unique=True)
            p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
            # assert len(rec_list[u]) > 0
            score += np.sum(p_at_k) / np.min([len(hidden_ratings[u]), len(rec_list[u])])
            i += 1
    return score / i if i != 0 else 0  # Really??


def compute_similarity_matrix(matrix, k, sh, row_wise=True, partition_size=1000):
    print "Computing similarity matrix"
    matrix = check_matrix(matrix, format='csr' if row_wise else 'csc')
    matrix_norms = matrix.copy()
    matrix_norms.data **= 2

    matrix_norms = matrix_norms.sum(axis=1 if row_wise else 0)

    matrix_norms = np.asarray(np.sqrt(matrix_norms)).ravel()
    matrix_norms += 1e-6
    repetitions = np.diff(matrix.indptr)

    matrix_norms = np.repeat(matrix_norms, repetitions)
    matrix.data /= matrix_norms
    sim = None
    if row_wise:
        n_iterations = matrix.shape[0] / partition_size + (matrix.shape[0] % partition_size != 0)
        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations
            start = i * partition_size
            end = start + partition_size if i < n_iterations - 1 else matrix.shape[0]
            partitioned_matrix = matrix[start:end, ]
            similarity_matrix = partitioned_matrix.dot(matrix.T).toarray().astype(np.float32)
            similarity_matrix[np.arange(similarity_matrix.shape[0]), np.arange(start,end)] = 0.0

            if sh > 0:
                similarity_matrix = apply_shrinkage(partitioned_matrix, matrix, similarity_matrix,sh)

            idx_sorted = np.argsort(similarity_matrix, axis=1)
            not_top_k = idx_sorted[:, :-k]
            similarity_matrix[np.arange(similarity_matrix.shape[0]), not_top_k.T] = 0.0

            # make it sparse again
            similarity_matrix = sps.csr_matrix(similarity_matrix)

            if i == 0:
                sim = similarity_matrix.copy()
                top_k_idx = idx_sorted[:,-k:]
            else:
                sim = sps.vstack([sim, similarity_matrix])
                top_k_idx = np.vstack((top_k_idx,idx_sorted[:,-k:]))

    return sim, top_k_idx


def compute_partial_similarity_matrix(matrix, sh, non_zero_cols, row_wise=True):
    print "Computing similarity matrix"
    matrix = check_matrix(matrix, format='csr' if row_wise else 'csc')
    matrix_norms = matrix.copy()
    matrix_norms.data **= 2

    matrix_norms = matrix_norms.sum(axis=1 if row_wise else 0)

    matrix_norms = np.asarray(np.sqrt(matrix_norms)).ravel()
    matrix_norms += 1e-6
    repetitions = np.diff(matrix.indptr)

    matrix_norms = np.repeat(matrix_norms, repetitions)
    matrix.data /= matrix_norms

    if row_wise:
        partitioned_matrix = matrix[non_zero_cols,:]
        partitioned_similarity_matrix = partitioned_matrix.dot(matrix.T).toarray().astype(np.float32)
        partitioned_similarity_matrix[np.arange(partitioned_similarity_matrix.shape[0]), non_zero_cols] = 0.0
        if sh > 0:
            partitioned_similarity_matrix = apply_shrinkage_partial(matrix, partitioned_similarity_matrix, non_zero_cols, sh)
        # make it sparse again
        partitioned_similarity_matrix = sps.csr_matrix(partitioned_similarity_matrix)
    partial_indptr = np.zeros_like(non_zero_cols, dtype=int)
    partial_indptr[non_zero_cols] = np.diff(partitioned_similarity_matrix.indptr)
    partial_indptr = np.cumsum(partial_indptr)
    partial_indptr = np.concatenate((np.array([0]), partial_indptr))
    partial_similarity_matrix = sps.csr_matrix((partitioned_similarity_matrix.data,
                                                partitioned_similarity_matrix.indices,
                                                partial_indptr),
                                               (matrix.shape[0], matrix.shape[0]))
    return partial_similarity_matrix


def apply_shrinkage_partial(matrix, partial_similarity_matrix, non_zero_cols, sh, row_wise=True):
    matrix_ind = matrix.copy()
    matrix_ind.data = np.ones_like(matrix_ind.data)

    if row_wise:
        co_counts = matrix_ind[non_zero_cols,:].dot(matrix_ind.T).toarray().astype(np.float32)

    co_counts /= (co_counts + sh)
    return partial_similarity_matrix * co_counts


def apply_shrinkage(partitioned_matrix, matrix, dist, sh, row_wise=True):
    partitioned_ind = partitioned_matrix.copy()
    partitioned_ind.data = np.ones_like(partitioned_ind.data)
    matrix_ind = matrix.copy()
    matrix_ind.data = np.ones_like(matrix_ind.data)

    if row_wise:
        co_counts = partitioned_ind.dot(matrix_ind.T).toarray().astype(np.float32)
    else:
        co_counts = matrix_ind.T.dot(partitioned_ind).toarray().astype(np.float32)

    co_counts /= (co_counts + sh)
    return dist * co_counts


# ICM in rec constructor?
def cv_search(rec, urm, icm, ignore_mask, sample_size=40000, sim_partition_size=1000, urm_partition_size=1000):
    np.random.seed(1)
    perm = np.random.permutation(urm.data.shape[0])[:-sample_size]
    urm_sample = urm.copy()
    urm_sample.data[perm] = 0.0
    urm_sample.eliminate_zeros()
    non_zero_rows = np.diff(urm_sample.indptr) != 0
    non_zero_cols = np.diff(urm_sample.tocsc().indptr) != 0
    urm_sample = urm_sample[non_zero_rows, :]
    urm_sample = urm_sample[:, non_zero_cols]
    icm_sample = icm[non_zero_cols, :]
    ignore_mask_sample = ignore_mask[non_zero_cols]
    print urm_sample.shape, icm_sample.shape

    #params = {'k': np.array([10, 20, 50, 100, 200, 500])[::-1], 'a_sh': np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])[::-1]}
    params = {'a_sh': np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50])[::-1]}
    grid = list(ParameterGrid(params))
    folds = 2
    kfold = KFold(n_splits=folds, shuffle=True)  # Shuffle UCM if necessary too, to keep indices correspondence
    splits = [(train, test) for train,test in kfold.split(urm_sample)]
    pred_ratings_perc = 0.75
    n = 5
    result = namedtuple('result', ['mean_score', 'std_dev', 'parameters'])
    results = []
    total = float(reduce(lambda acc, x: acc * len(x), params.itervalues(), 1) * folds)
    prog = 1.0
    prev_pars = None
    #max_k = np.max(params['k'])

    for pars in grid:
        rec = rec.set_params(**pars)
        maps = []
        print pars
        '''
        if prev_pars is None or prev_pars['a_sh'] != pars['a_sh']:
            sim_matrix, top_k_idx = compute_similarity_matrix(icm_sample, rec.k, rec.sh, partition_size=sim_partition_size)
            prev_pars = dict(pars)

        #sim_matrix = sim_matrix_max_k[np.arange(sim_matrix_max_k.shape[0]),top_k_idx[:,-rec.k:].T]
        if rec.k != max_k:
            sim_matrix[np.arange(sim_matrix.shape[0]),top_k_idx[:,-max_k:-rec.k].T] = 0.0
            sim_matrix.eliminate_zeros()
        '''

        for row_train, row_test in splits:
            #rec.fit(icm, sim=sim_matrix)
            rec.fit2(icm_sample)
            urm_test = urm_sample[row_test,]
            hidden_ratings = []
            for u in range(urm_test.shape[0]):
                relevant_u = urm_test[u,].nonzero()[1]  # Indices of rated items for test user u
                if len(relevant_u) > 2:#1 or 2
                    np.random.shuffle(relevant_u)
                    urm_test[u, relevant_u[int(len(relevant_u) * pred_ratings_perc):]] = 0
                    hidden_ratings.append(relevant_u[int(len(relevant_u) * pred_ratings_perc):])
                else:
                    hidden_ratings.append([])
            maps.append(map_scorer(rec, urm_test, hidden_ratings, n, ignore_mask=ignore_mask_sample, urm_partition_size=urm_partition_size))  # Assume rec to predict indices of items, NOT ids
            print "Progress: {:.2f}%".format((prog * 100) / total)
            prog += 1
        results.append(result(np.mean(maps), np.std(maps), pars))
        print "Result: ", result(np.mean(maps), np.std(maps), pars)
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP5", "Std"] + _.parameters.keys())
    print "Total scores: ", scores
    cols, col_feat, x_feat = 4, 'a_sh', 'k'
    f = sns.FacetGrid(data=scores, col=col_feat, col_wrap=cols, sharex=False, sharey=False)
    f.map(plt.plot, x_feat, 'MAP5')
    f.fig.suptitle("ItemCB CV MAP5 values")
    i_max, y_max = scores.MAP5.argmax(), scores.MAP5.max()
    i_feat_max = params[col_feat].index(scores[col_feat][i_max])
    f_max = f.axes[i_feat_max]
    f_max.plot(scores[x_feat][i_max], y_max, 'o', color='r')
    plt.figtext(0, 0, "With TF-IDF\nmaximum at (sh={:.5f},k={:.5f}, {:.5f}+/-{:.5f})".format(scores[col_feat][i_max],
                                                                                         scores[x_feat][i_max],
                                                                                         y_max,
                                                                                         scores['Std'][i_max]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    f.savefig('ItemCB CV MAP5 values.png', bbox_inches='tight')
    return scores


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
            maps.append(map_scorer(rec, URM_train[test_rows,], hidden_ratings,
                                   n))  # Assume rec to predict indices of items, NOT ids
            print "Progress: ", (prog * 100) / total, "%"
            prog += 1
        results.append(Result(np.mean(maps), np.std(maps), pars))
    scores = pd.DataFrame(data=[[_.mean_score, _.std_dev] + _.parameters.values() for _ in results],
                          columns=["MAP5", "Std"] + _.parameters.keys())
    cols, col_feat, x_feat = 3, 'sh', 'k'
    f = sns.FacetGrid(data=scores, col=col_feat, col_wrap=cols, sharex=False, sharey=False)
    f.map(plt.plot, x_feat, 'MAP5')
    f.fig.suptitle("ItemCB Holdout MAP5 values")
    i_max, y_max = scores.MAP5.argmax(), scores.MAP5.max()
    i_feat_max = params[col_feat].index(scores[col_feat][i_max])
    f_max = f.axes[i_feat_max]
    f_max.plot(scores[x_feat][i_max], y_max, 'o', color='r')
    plt.figtext(0, 0, "COMMENT\nmaximum at (sh={:.5f},k={:.5f}, {:.5f}+/-{:.5f})".format(scores[col_feat][i_max],
                                                                                         scores[x_feat][i_max],
                                                                                         y_max,
                                                                                         scores['Std'][i_max]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    f.savefig('ItemCB Holdout MAP5 values.png', bbox_inches='tight')
    return scores


def read_items():
    return pd.read_csv('../../inputs/item_profile.csv', sep='\t')


def read_interactions():
    ints = pd.read_csv('../../inputs/interactions_idx.csv', sep='\t')
    return sps.csr_matrix((ints['interaction_type'].values, (ints['user_idx'].values, ints['item_idx'].values)))


def check_matrix(x, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(x, sps.csc_matrix):
        return x.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(x, sps.csr_matrix):
        return x.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(x, sps.coo_matrix):
        return x.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(x, sps.dok_matrix):
        return x.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(x, sps.bsr_matrix):
        return x.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(x, sps.dia_matrix):
        return x.todia().astype(dtype)
    elif format == 'lil' and not isinstance(x, sps.lil_matrix):
        return x.tolil().astype(dtype)
    else:
        return x.astype(dtype)


def read_top_pops():
    aux = pd.read_csv('../../inputs/top_pop_ids.csv', sep='\t')['0'].values
    return aux


class ItemCB(BaseEstimator):
    def __init__(self, k=5, a_sh=2):  # k_rated??
        self.k = k
        self.sh = a_sh
        self.sim_mat = None

    def fit(self, icm, sim=None, partition_size=1000):
        if sim is None:
            self.sim_mat, _ = compute_similarity_matrix(icm, self.k, self.sh, partition_size=partition_size)
        else:
            self.sim_mat = sim

    def fit2(self, icm):
        self.icm = icm

    def predict2(self, urm, n, ignore_mask, partition_size=1000):
        print "Started prediction"
        user_profile = urm
        n_iterations = user_profile.shape[0] / partition_size + (user_profile.shape[0] % partition_size != 0)

        ranking = None
        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations

            start = i * partition_size
            end = start + partition_size if i < n_iterations - 1 else user_profile.shape[0]

            partitioned_profiles = user_profile[start:end, :]
            partition_items = np.diff(partitioned_profiles.tocsc().indptr) != 0
            print np.extract(partition_items == True, partition_items).shape[0]#, partitioned_profiles.shape
            self.sim_mat = compute_partial_similarity_matrix(self.icm,self.sh,partition_items)

            scores = partitioned_profiles.dot(self.sim_mat).toarray().astype(np.float32)
            print np.extract(scores != 0, scores).shape[0]

            # normalization
            rated = partitioned_profiles.copy()
            rated.data = np.ones_like(partitioned_profiles.data)
            den = rated.dot(self.sim_mat).toarray().astype(np.float32)
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den

            # remove the ones that are already rated
            nonzero_indices = partitioned_profiles.nonzero()
            scores[nonzero_indices[0], nonzero_indices[1]] = 0.0

            # remove the ignored ones
            scores[:, ignore_mask] = 0.0
            partition_ranking = scores.argsort()[:,::-1]
            partition_ranking = partition_ranking[:, -n:]  # leave only the top n

            if i == 0:
                ranking = partition_ranking.copy()
            else:
                ranking = np.vstack((ranking, partition_ranking))

        return ranking

    def predict(self, urm, n, ignore_mask, partition_size=1000):
        print "Started prediction"
        user_profile = urm
        n_iterations = user_profile.shape[0] / partition_size + (user_profile.shape[0] % partition_size != 0)

        ranking = None
        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations

            start = i * partition_size
            end = start + partition_size if i < n_iterations - 1 else user_profile.shape[0]

            partitioned_profiles = user_profile[start:end,:]
            scores = partitioned_profiles.dot(self.sim_mat.T).toarray().astype(np.float32)
            print np.extract(scores != 0, scores).shape[0], scores.shape[0]*scores.shape[1]

            # normalization
            rated = partitioned_profiles.copy()
            rated.data = np.ones_like(partitioned_profiles.data)
            den = rated.dot(self.sim_mat.T).toarray().astype(np.float32)
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den

            # remove the ones that are already rated
            nonzero_indices = partitioned_profiles.nonzero()
            scores[nonzero_indices[0], nonzero_indices[1]] = 0.0

            # remove the ignored ones
            scores[:, ignore_mask] = 0.0
            partition_ranking = scores.argsort()[::-1]
            partition_ranking = partition_ranking[:,:n]  # leave only the top n

            if i == 0:
                ranking = partition_ranking.copy()
            else:
                ranking = np.vstack((ranking, partition_ranking))

        return ranking


def generate_attrs(items_df):
    """ Generates normalized vectors from the item-content matrix, using
    TF-IDF. Normalization is computed among the values corresponding to the
    same attribute in the original item-content matrix.

    Arguments:
    items_df -- item-content matrix
    """
    attr_df = items_df.drop(['id', 'latitude', 'longitude', 'created_at', 'active_during_test'],
                            axis=1)
    attr_mats = {}
    to_dummies = ['career_level', 'country', 'region', 'employment']
    to_tfidf = ['discipline_id', 'industry_id']#, 'title', 'tags']

    attr_df['career_level'] = attr_df['career_level'].fillna(0)
    trans = CountVectorizer(token_pattern='\w+')
    for attr in to_dummies:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()

    trans = TfidfVectorizer(token_pattern='\w+')
    for attr in to_tfidf:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()

    return reduce(lambda acc, x: x if acc.shape == (1, 1) else sps.hstack([acc, x]), attr_mats.itervalues(),
                  sps.lil_matrix((1, 1)))



# Read items
items_dataframe = read_items()
URM = read_interactions()

actives = np.array(items_dataframe.active_during_test.values)
ignore_mask = actives == 0

item_ids = items_dataframe.id.values
ICM = generate_attrs(items_dataframe).tocsr()

test_users = pd.read_csv('../../inputs/target_users_idx.csv')['user_idx'].values
URM = URM[test_users,:]
recommender = ItemCB()
recommender.fit2(ICM)
recs = recommender.predict2(URM, 5, ignore_mask, partition_size=200)

# cv_search(recommender, URM, ICM, ignore_mask, sample_size=100000, sim_partition_size=5000, urm_partition_size=1000)
user_df = pd.read_csv('../../inputs/user_profile.csv', sep='\t')
out_file = open('../../output/ItemCB 15_11', 'wb')
out_file.write('user_id,recommended_items\n')
for i in range(len(recs)):
    out_file.write(str(user_df.loc[test_users[i]]['user_id']) + ',' + reduce(lambda acc, x: acc+str(x) + ' ', item_ids[recs[i]], '') + '\n')
out_file.close()