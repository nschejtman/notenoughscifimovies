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
    recs_fSLIM = pd.read_csv('../../output/Item fSLIM (Ridge) 2000sh 30000k 100000alpha ratings1.csv', sep=',')
    recs_SLIM = pd.read_csv('../../output/Item SLIM (Ridge) 100000alpha ratings1.csv', sep=',')
    recs_SCM_SVM = pd.read_csv('../../output/Item SCM CSVM 10minus2.csv', sep=',')
    final_recs = []
    final_recs_labels = []
    scores = np.array([1199, 1058, 842])
    p_fSLIM = scores[0] / scores.sum()
    p_SLIM = scores[1] / scores.sum()
    p_SCM_SVM = scores[2] / scores.sum()
    probs = np.array([p_fSLIM, p_SLIM, p_SCM_SVM])


    freqs = np.zeros((3, 5))
    for i in range(recs_SLIM.shape[0]):
        queue_fSLIM = deque(recs_fSLIM['recommended_items'][i].split())
        queue_SCM_SVM = deque(recs_SCM_SVM['recommended_items'][i].split())
        queue_SLIM = deque(recs_SLIM['recommended_items'][i].split())

        queues = [queue_fSLIM, queue_SLIM, queue_SCM_SVM]

        recs = []
        labels_labels = ["fSLIM", "SLIM", "SVM"]
        labels = []
        while len(recs) < 5:
            # prob_decay = 1
            # for i in range(4):
            #     probs[i] = probs_i[i] * (prob_decay ** (len(queues[i]) - 5))
            # probs = probs / np.sum(probs)

            p = np.random.rand()
            idx = bisect.bisect(probs.cumsum(), p) - 1

            item = queues[idx].popleft()

            if item not in recs:
                freqs[idx, len(recs)] += 1
                recs.append(item)
                labels.append(labels_labels[idx])

        final_recs.append(recs)
        final_recs_labels.append(labels)

    ut.write_recommendations('Hybrid_SM4', final_recs, recs_SLIM['user_id'].values)
    ut.write_recommendations("Hybrid SM4 LABELS", final_recs_labels, recs_SLIM['user_id'].values)


main()
