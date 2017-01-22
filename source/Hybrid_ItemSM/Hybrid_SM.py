import numpy as np
import scipy.sparse as sps
import pandas as pd
import sys
sys.path.append('./../')
import utils.utils as ut
from collections import deque

def main():
    np.random.seed(1)
    recs_SLIM = pd.read_csv('../../output/Item fSLIM (Ridge) 2000sh 20000k 100000alpha.csv', sep=',')
    recs_SCM = pd.read_csv('../../output/Item SCM CSVM 10minus2.csv', sep=',')
    p_SLIM = 0.5
    final_recs = []

    for i in recs_SLIM.shape[0]:
        queue_SLIM = deque(recs_SLIM['recommended_items'][i])
        queue_SCM = deque(recs_SCM['recommended_items'][i])
        recs = []
        while len(recs) < 5:
            item = (queue_SLIM.popleft() if np.random.rand() < p_SLIM else queue_SCM.popleft())
            if item not in recs:
                recs.append(item)

main()