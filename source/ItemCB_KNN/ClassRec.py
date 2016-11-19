import scipy.sparse as sps
import time
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression as LRC
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder


def cross_validate(X, Y, folds, test_size):
    penalties = ['l1', 'l2']
    Cs = np.linspace(1000, 10000,
                     num=10)  # np.linspace(1000, 10000, num=10)#np.linspace(100, 1000, num=10+1)#np.linspace(1, 100, num=10)
    solvers = ['newton-cg', 'lbfgs']  # ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    tols = [0.001, 0.0005, 0.0001]  # np.linspace(0.0001, 0.001, num=10)
    mult = ['multinomial']  # ['ovr', 'multinomial']
    out_file = open('cross_results.txt', 'wb')
    sss = StratifiedShuffleSplit(Y, n_iter=folds, test_size=test_size, random_state=0)
    solvers = ['lbfgs']
    tols = [0.0001]

    for s in solvers:
        for m in mult:
            for p in penalties:
                if (s in ['newton-cg', 'lbfgs', 'sag'] and p == 'l1') or (
                        s not in ['newton-cg', 'lbfgs'] and m == 'multinomial'):
                    continue
                for c in Cs:
                    for t in tols:
                        acc = 0
                        for i_train, i_test in sss:
                            clf = LRC(penalty=p, tol=t, C=c, solver=s, multi_class=m, max_iter=1000)
                            clf.fit(X.values[i_train], Y[i_train])
                            Y_t = clf.predict_proba(X.values[i_test])
                            acc += log_loss(Y[i_test], Y_t)
                        out_file.write(str(p) + ' ' + str(c) + ' ' + str(s) + ' ' + str(t) + ' ' + str(m) + ' ' + str(
                            acc / folds) + '\n')
                        print
                        p, c, s, t, m, acc / folds

    out_file.close()


def predict_test(X, Y, clf):
    clf.fit(X.values, Y.values)
    test = pd.read_csv('../data/test_norm1.csv')
    X_t = test[test.columns.difference(['id'])]
    Y_t = clf.predict_proba(X_t.values).astype(float)

    out_file = open("LogReg.csv", "wb")
    out_wr = csv.writer(out_file)
    out_wr.writerow(['id'] + clf.classes_.tolist())
    ids = test.id.tolist()
    for i in range(len(Y_t)):
        row = [ids[i]]
        row.extend([i for i in Y_t[i].tolist()])
        # row.extend(['{:0.10f}'.format(i) for i in Y_t[i].tolist()])
        out_wr.writerow(row)
    out_file.close()


def calculate_error_dist(X, Y, clf, folds, test_size):
    clss = dict(enumerate(Y.unique()))
    clss_dic = {c: i for i, c in clss.items()}
    Y = np.array([clss_dic[y] for y in Y.values])
    n_classes = len(clss)
    miss = np.zeros((n_classes, n_classes), dtype=np.float)
    sss = StratifiedShuffleSplit(Y, n_iter=folds, test_size=test_size, random_state=0)
    for i_train, i_test in sss:
        clf.fit(X.values[i_train], Y[i_train])
        Y_t = clf.predict_proba(X.values[i_test])
        for i in range(len(Y_t)):
            miss[Y[i_test[i]]][np.argmax(Y_t[i])] += 1
    miss = miss / folds
    ##    np.savetxt('error_dist.txt', miss, fmt= "%.3f")
    ##    out_file = open('classes.txt', 'wb')
    ##    out_file.write(str(clss_dic))
    ##    out_file.close()
    return miss, clss, clss_dic


def inspect_error_dist(miss, clss, clss_dic):
    out_file = open('error_dist1.txt', 'wb')
    for i in range(len(miss)):
        if miss[i][i] <= 1.8:
            print
            i
            print
            {j: miss[i][j] for j in range(len(miss)) if miss[i][j] != 0}
            out_file.write(str(i) + '\n')
            out_file.write(str({j: miss[i][j] for j in range(len(miss)) if miss[i][j] != 0}) + '\n')
    out_file.close()


def generate_attrs( items_df):
    ''' Generates normalized vectors from the item-content matrix, using
    TF-IDF. Normalization is computed among the values corresponding to the
    same attribute in the original item-content matrix.

    Arguments:
    items_df -- item-content matrix
    '''

    attr_df = items_df.drop(['id', 'latitude', 'longitude', 'created_at', 'active_during_test'],
                                 axis=1)

    to_dummies = ['career_level', 'country', 'region', 'employment']
    to_tfidf = ['discipline_id', 'industry_id', 'title', 'tags']

    attr_df['career_level'] = attr_df['career_level'].fillna(0)
    attr_df['title'] = attr_df['title'].fillna('NULL').values
    attr_df['tags'] = attr_df['tags'].fillna('NULL').values

    # Generate binary matrix
    #attr_df = pd.get_dummies(attr_df, columns=to_dummies)

    attr_mat = {_: generate_tfidf(attr_df[_].map(str).values) for _ in to_tfidf+to_dummies}
    attr_mat = reduce(lambda acc, x: x if acc.shape == (1,1) else sps.hstack([acc,x]), attr_mat.itervalues(), sps.lil_matrix((1,1))).tocsr()

    attr_df = attr_df.drop(to_tfidf, axis=1)
    return attr_mat


def generate_tfidf(data):
    vectorizer = CountVectorizer(token_pattern='\w+')
    trans = TfidfTransformer()
    tf = vectorizer.fit_transform(data)
    return trans.fit_transform(tf)  # , vectorizer.vocabulary_)


item_df = pd.read_csv('../../inputs/item_profile.csv', sep='\t')
all_ = generate_attrs(item_df)
print all_.shape
all_ = reduce(lambda acc, x: x if acc.shape == (1,1) else sps.vstack([acc,x]), [all_ for i in range(4)], sps.lil_matrix((1,1))).tocsr()
print all_.shape
X = all_
Y = [0 if i < X.shape[0]/2 else 1 for i in range(X.shape[0])]
# cross_validate(X, Y, 5, 0.2)
# Netwon and lbfgs multinomial have the best performance, being lbfgs slightly better.
# Sag is too slow, liblinear not too accurate. With C=100000 and tol = 0.0001 best results
# In submission the best is C=500, tol=0.001 and newron multinomial
print "Training..."
st = time.time()
clf = LRC(penalty='l2', tol=0.0001, C=500, solver='newton-cg')
clf.fit(X, Y)
et = time.time()
print "Train: ", et-st
st = time.time()
y_t = clf.predict(all_)
et = time.time()
print "Pred: ", et-st
print y_t
#miss, clss, clssdic = calculate_error_dist(X, Y, clf, 5, 0.2)
# inspect_error_dist(miss, clss, clssdic)
