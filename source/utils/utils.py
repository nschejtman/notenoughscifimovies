import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


def generate_icm_unified_attrs(items_df):
    attr_df = items_df.drop(['id', 'latitude', 'longitude', 'created_at', 'active_during_test'],
                            axis=1)

    attr_mats = {}
    attrs = ['career_level', 'country_region', 'employment', 'discipline_id', 'industry_id', 'title', 'tags']

    attr_df['career_level'] = attr_df['career_level'].fillna(0)
    attr_df['career_level'] = attr_df['career_level'].astype(dtype=int)

    country_dict = {c: i for i, c in enumerate(attr_df['country'].unique())}
    attr_df['country'] = attr_df['country'].apply(lambda x: country_dict[x])
    attr_df['country_region'] = attr_df['country'].apply(lambda x: x * 100) + attr_df['region']

    trans = CountVectorizer(token_pattern='\w+')
    for attr in attrs:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()

    icm = reduce(lambda acc, x: x if acc.shape == (1, 1) else sps.hstack([acc, x]), attr_mats.itervalues(),
                  sps.lil_matrix((1, 1))).tocsr()

    trans = TfidfTransformer()
    icm = trans.fit_transform(icm).tocsr()
    return icm


def urm_to_tfidf(urm):
    trans = TfidfTransformer()
    urm_tfidf = trans.fit_transform(urm).tocsr()
    return urm_tfidf


def generate_icm(items_df):
    """ Generates normalized vectors from the item-content matrix, using
    TF-IDF. Normalization is computed among the values corresponding to the
    same attribute in the original item-content matrix.

    Arguments:
    items_df -- item-content matrix
    """
    attr_df = items_df.drop(['id', 'latitude', 'longitude', 'created_at', 'active_during_test'],
                            axis=1)

    attr_mats = {}
    to_dummies = ['career_level', 'country_region', 'employment']#'country', 'region']
    to_tfidf = ['discipline_id', 'industry_id', 'title', 'tags']

    attr_df['career_level'] = attr_df['career_level'].fillna(0)
    attr_df['career_level'] = attr_df['career_level'].astype(dtype=int)

    country_dict = {c: i for i, c in enumerate(attr_df['country'].unique())}
    attr_df['country'] = attr_df['country'].apply(lambda x: country_dict[x])
    attr_df['country_region'] = attr_df['country'].apply(lambda x: x * 100) + attr_df['region']

    trans = CountVectorizer(token_pattern='\w+')
    for attr in to_dummies:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()

    trans = TfidfVectorizer(token_pattern='\w+')
    for attr in to_tfidf:
        attr_mats[attr] = trans.fit_transform(attr_df[attr].map(str).values).tocsr()

    return reduce(lambda acc, x: x if acc.shape == (1, 1) else sps.hstack([acc, x]), attr_mats.itervalues(),
                  sps.lil_matrix((1, 1))).tocsr()



def map_scorer(recommender, urm_test, hidden_ratings, n, non_active_items_mask, global_bias=None, item_bias=None, user_bias=None):
    score = 0
    if global_bias is not None:
        if item_bias is None and user_bias is None:
            rec_list = recommender.predict(urm_test, n, non_active_items_mask, global_bias, item_bias, user_bias)
        else:
            rec_list = recommender.predict(urm_test, n, non_active_items_mask, global_bias)
    else:
        rec_list = recommender.predict(urm_test, n, non_active_items_mask)
    i = 0
    for u in range(urm_test.shape[0]):
        if len(hidden_ratings[u]) > 0:
            is_relevant = np.in1d(rec_list[u], hidden_ratings[u], assume_unique=True)
            p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
            assert len(rec_list[u]) > 0
            score += np.sum(p_at_k) / np.min([len(hidden_ratings[u]), len(rec_list[u])])
            i += 1
    return score / i if i != 0 else 0



def read_items():
    items_df = pd.read_csv('../../inputs/item_profile.csv', sep='\t')
    items_df['career_level'] = items_df['career_level'].fillna(0)
    items_df['career_level'] = items_df['career_level'].astype(dtype=int)
    return items_df

def read_interactions():
    ints = pd.read_csv('../../inputs/interactions_idx.csv', sep='\t')
    return sps.csr_matrix((ints['interaction_type'].values, (ints['user_idx'].values, ints['item_idx'].values)))

def write_recommendations(name, recommendations, test_users_idx, item_ids):
    if item_ids is None:
        user_df = pd.read_csv('../../inputs/user_profile.csv', sep='\t')
        out_file = open('../../output/'+name+'.csv', 'wb')
        out_file.write('user_id,recommended_items\n')
        for i in range(len(recommendations)):
            out_file.write(str(test_users_idx[i]) + ',' + reduce((lambda acc, x: acc + str(x) + ' '), recommendations[i], '') + '\n')
        out_file.close()
    else:
        user_df = pd.read_csv('../../inputs/user_profile.csv', sep='\t')
        out_file = open('../../output/'+name+'.csv', 'wb')
        out_file.write('user_id,recommended_items\n')
        for i in range(len(recommendations)):
            out_file.write(str(user_df.loc[test_users_idx[i]]['user_id']) + ',' + reduce(lambda acc, x: acc + str(item_ids[x]) + ' ',
                                                                                     recommendations[i], '') + '\n')
        out_file.close()


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


def read_top_pops(count=False):
    if count:
        aux = None
    else:
        aux = pd.read_csv('../../inputs/top_pop_sum_idx.csv', sep='\t')['0'].values
    return aux


def normalize_matrix(matrix, row_wise=True):
    matrix = check_matrix(matrix, format='csr' if row_wise else 'csc')
    matrix_norms = matrix.copy()
    matrix_norms.data **= 2

    matrix_norms = matrix_norms.sum(axis=1 if row_wise else 0)

    matrix_norms = np.asarray(np.sqrt(matrix_norms)).ravel()
    matrix_norms += 1e-6
    repetitions = np.diff(matrix.indptr)

    matrix_norms = np.repeat(matrix_norms, repetitions)
    matrix.data /= matrix_norms
    return matrix


#Matrix assumed to be already normalized
def compute_similarity_matrix_knn(matrix, k, sh, row_wise=True, partition_size=1000):
    print "Computing similarity matrix"
    matrix = normalize_matrix(matrix, row_wise)
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
                break
            else:
                sim = sps.vstack([sim, similarity_matrix])
                top_k_idx = np.vstack((top_k_idx,idx_sorted[:,-k:]))

    return sim, top_k_idx


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


def compute_similarity_matrix_mask(matrix, sh, batch_mask, row_wise=True): # TODO: column_wise
    print "Computing similarity matrix - batch"
    # TODO: normalize only once
    matrix = normalize_matrix(matrix, row_wise)

    if row_wise:
        partitioned_matrix = matrix[batch_mask, :]
        partitioned_similarity_matrix = partitioned_matrix.dot(matrix.T).toarray().astype(np.float32)
        partitioned_similarity_matrix[np.arange(partitioned_similarity_matrix.shape[0]), batch_mask] = 0.0
        if sh > 0:
            partitioned_similarity_matrix = apply_shrinkage_batch(matrix, partitioned_similarity_matrix, batch_mask, sh)
        # make it sparse again
        partitioned_similarity_matrix = sps.csr_matrix(partitioned_similarity_matrix)
    partial_indptr = np.zeros_like(batch_mask, dtype=int)
    partial_indptr[batch_mask] = np.diff(partitioned_similarity_matrix.indptr)
    partial_indptr = np.cumsum(partial_indptr)
    partial_indptr = np.concatenate((np.array([0]), partial_indptr))
    partial_similarity_matrix = sps.csr_matrix((partitioned_similarity_matrix.data,
                                                partitioned_similarity_matrix.indices,
                                                partial_indptr),
                                               (matrix.shape[0], matrix.shape[0]))
    return partial_similarity_matrix


def apply_shrinkage_batch(matrix, partial_similarity_matrix, batch_mask, sh, row_wise=True):
    matrix_ind = matrix.copy()
    matrix_ind.data = np.ones_like(matrix_ind.data)

    if row_wise:
        co_counts = matrix_ind[batch_mask, :].dot(matrix_ind.T).toarray().astype(np.float32)

    co_counts /= (co_counts + sh)
    return partial_similarity_matrix * co_counts



def produce_sample(urm, icm, ucm, non_active_items_mask, sample_size, sample_from_urm, item_bias=None, user_bias=None): # TODO: Add sample on users
    if sample_from_urm:
        perm = np.random.permutation(urm.shape[0])[:sample_size]
        urm_sample = urm[perm,]
        if item_bias is None and user_bias is None:
            return urm_sample, icm, None, non_active_items_mask
        else:
            user_bias_sample = user_bias[perm]
            return urm_sample, icm, None, non_active_items_mask, item_bias, user_bias_sample
    else:
        interactions_to_remove = np.random.permutation(urm.data.shape[0])[:-sample_size]
        urm_sample = urm.copy()
        urm_sample.data[interactions_to_remove] = 0.0
        urm_sample.eliminate_zeros()
        retained_users = np.diff(urm_sample.indptr) != 0
        retained_items = np.diff(urm_sample.tocsc().indptr) != 0
        urm_sample = urm_sample[retained_users, :]
        urm_sample = urm_sample[:, retained_items]
        icm_sample = icm[retained_items, :]
        non_active_items_mask_sample = non_active_items_mask[retained_items]
        return urm_sample, icm_sample, None, non_active_items_mask_sample


def global_effects(urm):
    urm = sps.csc_matrix(urm, dtype=np.float32, copy=False)
    global_bias = np.mean(urm.data)
    urm.data -= global_bias

    urm = urm.tocsc(copy=False)
    item_bias = np.asarray(urm.sum(axis=0)).ravel()
    dens = np.diff(urm.indptr)
    dens[dens == 0] = 1
    item_bias /= dens
    for i in range(len(urm.indptr) - 1):
        urm.data[urm.indptr[i]:urm.indptr[i + 1]] -= item_bias[i]

    urm = urm.tocsr(copy=False)
    user_bias = np.asarray(urm.sum(axis=1)).ravel()
    dens = np.diff(urm.tocsr().indptr)
    dens[dens == 0] = 1
    user_bias /= dens
    for i in range(len(urm.indptr) - 1):
        urm.data[urm.indptr[i]:urm.indptr[i + 1]] -= user_bias[i]

    return urm, global_bias, item_bias, user_bias


