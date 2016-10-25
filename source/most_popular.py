import numpy as np
import pandas as pd
import operator
import csv

data = pd.read_csv('~/Desktop/interactions.csv', sep = '\t')
dic = {x:0 for x in data.item_id.unique()}

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
     test_users = pd.read_csv('~/Desktop/target_users.csv')
     for x in test_users.itertuples():
          out_file.write(str(x.user_id) + ',' + most_popular_str + '\n')
     out_file.close()

write_csv('most_popular')
