"""
Kaggle : StumbleUpon Evergreen Classification Challenge
Author : log0
Contact : im __dot__ ckieric __at__ gmail __dot__ com

============================================================================================
Comments:

This is my 13th submission, though I did not select this into my final private leaderboard.
This is actually my best submission, which bested the other two I selected. This is a good
example of how overfitting can pull you down. I hope this will serve to me and the others
a good lesson of how higher dimension engineered features could be hazardous, potentially.

CV score: 0.888209675529
Public score: 0.88415 (Rank 57)
Private score: 0.87967 (Rank 291)

Note that due to the complexity and messiness of the code, this does not run by itself. But
the gist of the solution logic is here.

Enjoy!

============================================================================================
Solution Description:

First, I manually processed the body, title and url and use a custom word tokenizer on
normalized text which then I feed into a TfidfVectorizer. Along with it, I engineered 
some features from the document contents via analyzing sentence POS tags. On top of it, 
and this is where I made my demise... I engineered some higher dimensional features which
 is a perfect recipe for overfitting (which the CV did not tell). Lastly, I applied a
 Latent Semantic Analysis (LSA) with (400-500 components) then used an LogisticRegression.
 along with TfidfVectorizer (carefully not looking into the test features, of course).

============================================================================================
Log output:

2013-10-31 14:41:19.118000
Reading 10000 data
Max data read : 7395
Max data read : 3171
Loaded from cache
bestwords count = 58088
train vector ready
Final X shape : (7395, 58143)
submission vector ready
2013-10-31 14:48:58.784000
0.888209675529 | [ 0.89300066  0.88945616  0.8989167   0.8817395   0.87793536]
2013-10-31 14:55:36.726000
============================================================================================

"""

import os
import re
import csv
import json
import numpy as np
import pickle
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import *
from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from text.blob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords
import collections, itertools
import nltk.classify.util, nltk.metrics
import scipy.sparse as sparse

import data_io
import features

r_nonalnum = re.compile(r'\W+')
r_nonans   = re.compile(r'[^\w\s]+')
r_digit    = re.compile(r'\d+')
r_url      = re.compile(r'https?://')
r_nonword  = re.compile(r'[A-Z\d]+')

def get_bestwords(contents, labels, limit = 10000, n = None, cache = True):
    if cache:
        if n:
            cache_path = 'cache/%s_%s.pkl' % (limit, n)
            if os.path.exists(cache_path):
                bestwords = pickle.load(open(cache_path, 'r'))
                print 'Loaded from cache'
                print 'bestwords count = %d' % (len(bestwords))
                return bestwords
    
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    
    pos_contents = contents[labels == 1]
    neg_contents = contents[labels != 0]
    
    pos_words = set()
    neg_words = set()
    
    for pos_content in pos_contents:
        pos_words = pos_words.union(word_tokenize(pos_content))
    
    for neg_content in neg_contents:
        neg_words = neg_words.union(word_tokenize(neg_content))
    
    for word in pos_words:
        word_fd.inc(word.lower())
        label_word_fd['pos'].inc(word.lower())
    
    for word in neg_words:
        word_fd.inc(word.lower())
        label_word_fd['neg'].inc(word.lower())
    
    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    
    word_scores = {}
    
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
            (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    
    best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:limit]
    bestwords = set([w for w, s in best])
    
    print 'all words count = %d' % (len(word_scores))
    print 'bestwords count = %d' % (len(bestwords))
    
    if cache:
        if n:
            cache_path = 'cache/%s_%s.pkl' % (limit, n)
            f = open(cache_path, 'w')
            pickle.dump(bestwords, f)
            print 'Dumped to cache'
    
    return bestwords

def get_doc_contents(data):
    contents = []
    blobs = [ json.loads(i) for i in data ]
    for i in blobs:
        row = {'title': '', 'body': '', 'url': ''}
        
        for key in row.keys():
            if i.has_key(key) and i[key]:
                row[key] = i[key]
        
        content = ' '.join([ i for i in row.itervalues() ])
        contents.append(content)
    
    return np.array(contents)

def scale_vector(v):
    return v / float(np.max(v))

def get_char_count_vector(contents, char):
    v = [ content.count(char) + 1 for content in contents ]
    return np.array(v)

r_digits = re.compile('\d')
def get_digit_count_vector(contents):
    global r_digits
    v = [ len(r_digits.findall(content)) for content in contents ]
    return np.array(v)

def get_meta_vector_2(contents_dict):
    v = []
    
    contents = [ content['body'] or '' for content in contents_dict ]
    
    def count_sent(s):
        if s != None:
            return len(sent_tokenize(s))
        else:
            return 0
    v_count_sent = np.vectorize(count_sent)
    
    def count_url(s):
        return len(r_url.findall(s))
    v_count_url = np.vectorize(count_url)
    
    ws = {}
    ws['sent_count'] = v_count_sent(contents)
    ws['url_count'] = v_count_url(contents)
    ws['img_count'] = np.array([int(content['img_count'] or 0) for content in contents_dict])
    ws['?'] = get_char_count_vector(contents, '?').astype(float)
    ws['!'] = get_char_count_vector(contents, '!').astype(float)
    ws['.'] = get_char_count_vector(contents, '.').astype(float)
    ws[','] = get_char_count_vector(contents, ',').astype(float)
    
    v.append(ws['!']) # good stuff
    v.append(ws['.'] ** 2) # good stuff
    
    v = [scale_vector(i) for i in v]
    
    return np.array(v).transpose()

def get_meta_vector(tags, contents):
    v = []
    
    ws = {}
    ws['nn_vector'] = get_pos_tag_count(tags, ['NN', 'NNS', 'NNP', 'NNPS']).astype(float) # 1.0, 0.5
    ws['ad_vector'] = get_pos_tag_count(tags, ['JJ', 'JJR', 'JJS']).astype(float) # 1.0, 0.5
    ws['av_vector'] = get_pos_tag_count(tags, ['RB', 'RBR', 'RBS']).astype(float) # 1.0, 0.5,
    ws['vb_vector'] = get_pos_tag_count(tags, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']).astype(float) # 1.0, 0.5, 2.0
    ws['wd_vector'] = get_pos_tag_count(tags, ['WDT', 'WP', 'WP$', 'WRB']).astype(float) # 0.5
    ws['po_vector'] = get_pos_tag_count(tags, ['PRP', 'PRP$']).astype(float) # 1.0, 0.5
    ws['dt_vector'] = get_pos_tag_count(tags, ['DT']).astype(float)
    ws['cd_vector'] = get_pos_tag_count(tags, ['CD']).astype(float)
    ws['m1_vector'] = get_pos_tag_count(tags, ['DT', 'IN', 'JJ']).astype(float)
    ws['in_vector'] = get_pos_tag_count(tags, ['IN']).astype(float)
    ws['md_vector'] = get_pos_tag_count(tags, ['MD']).astype(float)
    ws['cn_vector'] = get_count_vector(tags).astype(float)
    ws['?'] = get_char_count_vector(contents, '?').astype(float)
    ws['!'] = get_char_count_vector(contents, '!').astype(float)
    ws['.'] = get_char_count_vector(contents, '.').astype(float)
    ws[','] = get_char_count_vector(contents, ',').astype(float)
    ws['numbers'] = get_digit_count_vector(contents).astype(float)
    
    v.append(ws['nn_vector']) # good stuff
    v.append(ws['ad_vector']) # good stuff
    v.append(ws['vb_vector']) # good stuff
    v.append(ws['po_vector']) # good stuff
    v.append(ws['cn_vector']) # good stuff
    v.append(ws['?']) # good stuff
    
    v.append(ws['nn_vector'] ** 0.5) # good stuff
    v.append(ws['ad_vector'] ** 0.5) # good stuff
    v.append(ws['av_vector'] ** 0.5) # good stuff
    v.append(ws['vb_vector'] ** 0.5) # good stuff
    v.append(ws['wd_vector'] ** 0.5) # good stuff
    v.append(ws['po_vector'] ** 0.5) # really good shit
    v.append(ws['dt_vector'] ** 0.5) # really good shit, pushes up to 0.88404
    v.append(ws['cn_vector'] ** 0.5) # good stuff
    v.append(ws['m1_vector'] ** 0.5) # good stuff
    
    v.append(ws['vb_vector'] ** 2) # good stuff
    
    v.append(ws['ad_vector'] ** 3.0) # good stuff
    v.append(ws['av_vector'] ** 3.0) # good stuff
    v.append(ws['vb_vector'] ** 3.0) # good stuff
    v.append(ws['cd_vector'] ** 3.0) # good stuff
    
    v.append(ws['nn_vector'] * ws['ad_vector']) # good stuff
    v.append(ws['nn_vector'] * ws['dt_vector']) # good stuff
    v.append(ws['nn_vector'] * ws['m1_vector']) # good stuff
    v.append(ws['nn_vector'] * ws['md_vector']) # good stuff
    v.append(ws['nn_vector'] * ws['?']) # good stuff
    v.append(ws['ad_vector'] * ws['av_vector']) # good stuff
    v.append(ws['ad_vector'] * ws['vb_vector']) # good stuff
    v.append(ws['ad_vector'] * ws['wd_vector']) # good stuff
    v.append(ws['ad_vector'] * ws['m1_vector']) # good stuff
    v.append(ws['ad_vector'] * ws['cn_vector']) # good stuff
    v.append(ws['av_vector'] * ws['wd_vector']) # good stuff
    v.append(ws['av_vector'] * ws['?']) # good stuff
    v.append(ws['av_vector'] * ws['!']) # good stuff
    v.append(ws['av_vector'] * ws['.']) # really good stuff
    v.append(ws['vb_vector'] * ws['wd_vector']) # good stuff
    v.append(ws['vb_vector'] * ws['?']) # good stuff
    v.append(ws['vb_vector'] * ws['!']) # good stuff
    v.append(ws['vb_vector'] * ws['.']) # good stuff
    v.append(ws['wd_vector'] * ws['po_vector']) # good stuff
    v.append(ws['wd_vector'] * ws['dt_vector']) # good stuff
    v.append(ws['po_vector'] * ws['?']) # good stuff
    v.append(ws['dt_vector'] * ws['cn_vector']) # good stuff
    v.append(ws['cd_vector'] * ws[',']) # good stuff
    v.append(ws['m1_vector'] * ws['cn_vector']) # no effect!
    v.append(ws['m1_vector'] * ws['?']) # good stuff
    v.append(ws['m1_vector'] * ws['numbers']) # good stuff
    v.append(ws['in_vector'] * ws['?']) # good stuff
    v.append(ws['in_vector'] * ws['numbers']) # good stuff
    v.append(ws['md_vector'] * ws['?']) # good stuff
    v.append(ws['md_vector'] * ws['numbers']) # good stuff
    v.append(ws['cn_vector'] * ws['!']) # good stuff
    v.append(ws['?'] * ws['.']) # good stuff
    v.append(ws[','] * ws['numbers']) # good stuff
    
    v = [scale_vector(i) for i in v]
    
    return np.array(v).transpose()

def get_pos_tag_count(tags, target):
    v = [ None for i in xrange(len(tags)) ]
    for i, tag in enumerate(tags):
        entries = (tag['body'] or []) + (tag['title'] or []) + (tag['url'] or [])
        v[i] = sum(1 if entry[1] in target else 0 for entry in entries)
        v[i] = v[i] + 1
    return np.array(v)

def get_count_vector(tags):
    count_vector = [ 0 for i in xrange(len(tags)) ]
    for i, tag in enumerate(tags):
        count_body = len(tag['body'] or [])
        count_title = len(tag['title'] or [])
        count_url = len(tag['url'] or [])
        count_vector[i] = count_body + count_title + count_url + 1
        
    return np.array(count_vector)

if __name__ == '__main__':
    print datetime.datetime.now()
    
    np.random.seed(1500) # 500 is original

    train_path = 'data/train.tsv'
    submission_path = 'data/test.tsv'
    
    n = 10000
    tags = pickle.load(open('cache/tagged.%s.pkl' % (n)))
    custom_contents = np.array(pickle.load(open('cache/custom_contents.%s.pkl' % (n))))
    submission_custom_contents = np.array(pickle.load(open('cache/s.custom_contents.pkl')))
    submission_tags = pickle.load(open('cache/s.tagged.pkl'))
    
    print 'Reading %s data' % (n)
    data = data_io.read_data(train_path, n)
    submission_data = data_io.read_data(submission_path, 10000) # use all
    
    contents = get_doc_contents(data[:, 2])
    contents = [ features.normalize(content) for content in contents ]
    
    Y = data[:, -1].astype(int)
    
    bestwords = get_bestwords(contents, Y, 100000, n)
    
    submission_contents = get_doc_contents(submission_data[:, 2])
    submission_contents = [ features.normalize(submission_content) for submission_content in submission_contents ]
    X_submission_ids = submission_data[:, 1]
    
    v = TfidfVectorizer(min_df = 2, binary = True, norm = 'l2', smooth_idf = True, sublinear_tf = True, strip_accents = 'unicode', vocabulary = bestwords, ngram_range = (1,2))
    X = v.fit_transform(contents)
    X_submission = v.transform(submission_contents)
    
    del data
    del submission_data
    
    meta_vector = get_meta_vector(tags, contents)
    del tags
    del contents
    
    meta_vector_2 = get_meta_vector_2(custom_contents)
    del custom_contents
    
    X = sparse.hstack((X, meta_vector, meta_vector_2))
    
    print 'train vector ready'
    print 'Final X shape : %s' % (str(X.shape))
    
    submission_meta_vector = get_meta_vector(submission_tags, submission_contents)
    del submission_tags
    del submission_contents
    
    submission_meta_vector_2 = get_meta_vector_2(submission_custom_contents)
    del submission_custom_contents
    
    X_submission = sparse.hstack((X_submission, submission_meta_vector, submission_meta_vector_2))
    
    print 'submission vector ready'
    print datetime.datetime.now()
    
    c = Pipeline([
        ('fe', TruncatedSVD(n_components = 400)),
        ('clf', LogisticRegression(tol = 0.0001, dual = True, penalty = 'l2')),
    ])
    
    cv_scores = cross_val_score(c, X, Y, cv = 5, scoring = 'roc_auc')
    print '%s | %s' % (np.mean(cv_scores), cv_scores)
    
    c.fit(X, Y)
    Y_submission_pred = c.predict_proba(X_submission)[:, 1]
    f = open('submission/simple_20.csv', 'w')
    f.write("urlid,label\n")
    for i in xrange(len(X_submission_ids)):
        url_id = X_submission_ids[i]
        label = Y_submission_pred[i]
        f.write("%s,%s\n" % (url_id, label))
    f.close()
    
    print datetime.datetime.now()