import numpy as np
import scipy.sparse as sps
import pandas as pd
import sys
sys.path.append('./../')
import utils.utils as ut
from collections import deque
import bisect
import time

def main():
    np.random.seed(4294967295 - 1239874)
    recs_fSLIM = pd.read_csv('../../output/Item fSLIM (Ridge) 2000sh 20000k 100000alpha.csv', sep=',')
    recs_SCM_SVM = pd.read_csv('../../output/Item SCM CSVM 10minus2.csv', sep=',')
    recs_SLIM = pd.read_csv('../../output/ItemSLIM (Ridge) Alpha1000000.csv', sep=',')
    recs_SCM_LR = pd.read_csv('../../output/Item SCM Clog_reg 20minus1.csv', sep=',')
    p_SLIM = 0.35
    final_recs = []
    final_recs_labels = []
    p_fSLIM = 0.27
    p_SLIM = 0.26
    p_SCM_SVM = 0.24
    p_SCM_LR = 0.23
    probs = np.array([p_fSLIM, p_SLIM, p_SCM_SVM, p_SCM_LR])
    probs_i = probs.copy()

    freqs = np.zeros((4, 5))
    for i in range(recs_SLIM.shape[0]):
        queue_fSLIM = deque(recs_fSLIM['recommended_items'][i].split())
        queue_SCM_SVM = deque(recs_SCM_SVM['recommended_items'][i].split())
        queue_SLIM = deque(recs_SLIM['recommended_items'][i].split())
        queue_SCM_LR = deque(recs_SCM_LR['recommended_items'][i].split())

        queues = [queue_fSLIM, queue_SLIM, queue_SCM_SVM, queue_SCM_LR]

        recs = []
        labels_labels = ["fSLIM", "SLIM", "SVM", "LR"]
        labels = []
        while len(recs) < 5:
            for i in range(4):
                probs[i] = probs_i[i] * (1.5 ** (len(queues[i])-5))
            probs = probs / np.sum(probs)

            p = np.random.rand()
            idx = bisect.bisect(probs.cumsum(), p)

            item = queues[idx].popleft()

            if item not in recs:
                freqs[idx, len(recs)] += 1
                recs.append(item)
                labels.append(labels_labels[idx])


        final_recs.append(recs)
        final_recs_labels.append(labels)


    ut.write_recommendations('Hybrid_SM3', final_recs, recs_SLIM['user_id'].values)
    ut.write_recommendations("Hybrid SM3 LABELS", final_recs_labels, recs_SLIM['user_id'].values)







main()