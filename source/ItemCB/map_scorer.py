##def map_scorer(rec, URM, items_ids):
##    score = 0
##    rec_list = est.predict(URM)
##    for u in URM.shape[0]:
##        is_relevant = np.in1d(rec_list[u], item_ids[u], assume_unique=True)
##        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
##        score += np.sum(p_at_k) / np.min([len(item_ids), len(rec_list)])
##    return score / URM.shape[0]
##

import pandas as pd
import numpy as np

data = pd.read_csv('../../inputs/interactions.csv', sep='\t')
item_df = pd.read_csv('../../inputs/item_profile.csv', sep='\t')
id_dic = {item_df.loc[i]['id']:i for i in range(item_df.shape[0])}
user_df = pd.read_csv('../../inputs/user_profile.csv', sep='\t')
us_dic = {user_df.loc[i]['user_id']:i for i in range(user_df.shape[0])}
data['item_id'] = data['item_id'].apply(lambda x: id_dic[x])
data['user_id'] = data['user_id'].apply(lambda x: us_dic[x])
grp = data.groupby(['user_id', 'item_id'])['interaction_type'].agg([np.max])
idxs = grp.index.values
out_df = pd.DataFrame([[idxs[i][0], idxs[i][1], grp.loc[idxs[i][0], idxs[i][1]]['amax']] for i in range(len(idxs))], columns=['user_idx', 'item_idx', 'interaction_type'])
out_df.to_csv('../../inputs/interactions_idx.csv', sep='\t', index=False)
