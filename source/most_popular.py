import numpy as np
import pandas as pd
import operator
import csv
import scipy.sparse as sps


data = pd.read_csv('../inputs/interactions.csv', sep = '\t')
dic = {x:0 for x in data.item_id.unique()}
items_df = pd.read_csv('../inputs/item_profile.csv', sep='\t')
# user_df = pd.read_csv('../inputs/user_profile.csv', sep='\t')
# us_dic = {user_df.loc[i]['user_id']:i for i in range(user_df.shape[0])}
# test_users = pd.read_csv('../inputs/target_users.csv')
# test_users['user_id'] = test_users['user_id'].apply(lambda x: us_dic[x])
# test_users.to_csv('../inputs/target_users_idx.csv', index=False)

def most_popular() :
     for row in data.itertuples() :
          dic[row.item_id] += row.interaction_type
     tops = sorted(dic.items(), key = operator.itemgetter(1), reverse=True)[0:5]
     return tops

print most_popular()

def write_csv(name) :
     most_popular_str = ""
     for x in most_popular():
          most_popular_str += str(x[0]) + " "
     # most_popular_str = most_popular_str[0, ]
     out_file = open(name + '.csv', 'wb')
     out_file.write('user_id,recommended_items\n')
     test_users = pd.read_csv('../inputs/target_users.csv')
     for x in test_users.itertuples():
          out_file.write(str(x.user_id) + ',' + most_popular_str + '\n')
     out_file.close()

# write_csv('most_popular')

def map_scorer ():
   print 'check 1'
   score = 0
   rec_list = most_popular()
   test_users_indices = pd.read_csv('../inputs/target_users_idx.csv')
   URM = read_interactions()
   tup = URM.nonzero()
   print 'check 2'
   aux = map(lambda x: items_df.loc[x]['id'], tup[1])
   n_users = len(aux)
   items_ids = [[] for _ in range(URM.shape[0])]
   print 'check 3'
   for i in range(n_users):
      items_ids[tup[0][i]].append(aux[i])
      # print item_ids[i]

   i = 0
   for u in test_users_indices['user_id'].values :
      if len(items_ids[u]) > 0 :
         is_relevant = np.in1d(rec_list, items_ids[u], assume_unique=True)
         p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

         assert len(rec_list) > 0
         score += np.sum(p_at_k) / np.min([len(items_ids[u]), len(rec_list)])
         i += 1
   print score / i
   return score / i

def read_interactions():
    ints = pd.read_csv('../inputs/interactions_idx.csv', sep='\t')
    return sps.csr_matrix((ints['interaction_type'].values, (ints['user_idx'].values, ints['item_idx'].values)))

map_scorer()
