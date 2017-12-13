import numpy as np
import matplotlib.pyplot as plt
import json
import re
import time
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, silhouette_samples, silhouette_score, \
    accuracy_score, make_scorer
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import kernel_ridge


def save_model(model, filename):
    with open(filename, 'w+') as file:
        pickle.dump(model, file)
    return model


def load_model(filename):
    with open(filename, 'r') as file:
        model = pickle.load(file)
    return model


def load_json(filename='trump_metadata.json', all_sources=True, strip_urls=True):
    with open(filename) as fd:
        data = json.load(fd)

    if not all_sources:
        data = [d for d in data if d['source'] == 'Twitter for Android']

    data_sanitized = data

    if not strip_urls:
        return data_sanitized

    for (idx, d) in enumerate(data):
        m = re.search('(https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]'
                      '\\.[^\\s]{2,}|www\\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]'
                      '\\.[^\\s]{2,}|https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9]'
                      '\\.[^\\s]{2,}|www\\.[a-zA-Z0-9]\\.[^\\s]{2,})', d['text'])
        if m is None:
            continue
        data_sanitized[idx]['text'] = d['text'][:m.start()] + d['text'][m.end():]

    return data_sanitized


def load_data(filename='trump_metadata.json', all_sources=True, strip_urls=True, text_only=False):
    data = load_json(filename, all_sources, strip_urls)
    if text_only:
        return [d['text'] for d in data]
    return data


def tweet_tokenizer(string):
    return re.findall('@[\w]+|#[\w]+|[0-9]+\.[0-9]+%?|[\w\']+|[!?]|[\'\"]', string)


def build_preprocessor_pipeline(ng_start=1, ng_end=4, truncate=True, tfidf=True, hashing=True, ncomp=100):
    steps = []
    if hashing:
        vectorizer = HashingVectorizer(analyzer='word', tokenizer=tweet_tokenizer, ngram_range=(ng_start, ng_end))
        steps += [('vectorize', vectorizer)]
        if tfidf:
            steps += [('transform', TfidfTransformer(use_idf=True, sublinear_tf=True))]
    else:
        if tfidf:
            vectorizer = TfidfVectorizer(analyzer='word', tokenizer=tweet_tokenizer, ngram_range=(ng_start, ng_end),
                                         lowercase=False, sublinear_tf=True, use_idf=True)
        else:
            vectorizer = CountVectorizer(analyzer='word', tokenizer=tweet_tokenizer, ngram_range=(ng_start, ng_end),
                                         lowercase=False)
        steps += [('vectorize', vectorizer)]
    if truncate:
        steps += [('svd', TruncatedSVD(n_components=ncomp))]

    return Pipeline(steps)


def build_kmeans_model(tweets, preprocessor, n_clusters=list([3])):
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    scores = [0] * len(n_clusters)

    for idx, k in enumerate(n_clusters):
        km = KMeans(n_clusters=k, init='k-means++')
        model = Pipeline([('preprocess', preprocessor), ('cluster', km)])
        start = time.time()
        model.fit(tweets)
        end = time.time()
        print 'Finished in %1.2fs!' % (end - start)

        scores[idx] = silhouette_score(preprocessor.transform(tweets), model.predict(tweets))
        save_model(model, 'kMeans%d_n%d.model' % (k, N))
        print scores[idx]

    return scores


def save_preprocess_data(preprocessor):
    import scipy.io as sio

    data = load_data(all_sources=True, strip_urls=True)
    tweets = [d['text'] for d in data]
    retweets = [d['retweet_count'] for d in data]


    sources = list(set([d['source'] for d in data]))

    android_idx = [idx for (idx, d) in enumerate(data) if d['source'] == 'Twitter for Android']
    iphone_idx = [idx for (idx, d) in enumerate(data) if d['source'] == 'Twitter for iPhone']
    other_idx = [idx for (idx, d) in enumerate(data)
                 if d['source'] != 'Twitter for Android' and d['source'] != 'Twitter for iPhone']

    X = preprocessor.fit_transform(tweets)
    sio.savemat('tweet_data.mat', {'X': X,
                                   'android_idx': android_idx,
                                   'iphone_idx': iphone_idx,
                                   'other_idx': other_idx,
                                   'retweets': retweets})


def build_linear_model(preprocessor):
    data = load_data(all_sources=True, strip_urls=True)
    android_data = [d for d in data if d['source'] == 'Twitter for Android']
    iphone_data = [d for d in data if d['source'] == 'Twitter for iPhone']

    print len(android_data), len(iphone_data)

    all_data = android_data + iphone_data
    np.random.shuffle(all_data)
    labels = [1 if d['source'] == 'Twitter for Android' else 0 for d in all_data]
    X = [d['text'] for d in all_data]

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=0)
    X_android = [x for c, x in zip(y_test, X_test) if c == 1]
    X_iphone = [x for c, x in zip(y_test, X_test) if c == 0]
    print len(X_android), len(X_iphone)

    classifier = GridSearchCV(linear_model.LogisticRegression(), {'C': np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])},
                              scoring=make_scorer(accuracy_score), cv=3)
    model = Pipeline([('preprocess', preprocessor), ('classifier', classifier)])
    model.fit(X_train, y_train)
    scores = model.named_steps['classifier'].cv_results_['mean_test_score']

    print 1.0 / len(X_test) * sum(model.predict(X_test) != y_test)
    print 1.0 / len(X_android) * sum(model.predict(X_android) == 0)
    print 1.0 / len(X_iphone) * sum(model.predict(X_iphone) == 1)

    plt.plot(np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]), 1 - scores)
    plt.xlabel('$\lambda$')
    plt.ylabel('Validation Error')
    plt.savefig('cv_lr.png')
    plt.close()

    return model


if __name__ == '__main__':
    # build_linear_model(build_preprocessor_pipeline(truncate=False, tfidf=True, hashing=False))

    #save_preprocess_data(build_preprocessor_pipeline(truncate=False, tfidf=True, hashing=False))
    #build_linear_model(build_preprocessor_pipeline(truncate=False, tfidf=True, hashing=False))
    #build_linear_model(build_preprocessor_pipeline(truncate=True, ncomp=100, tfidf=True, hashing=False))

    """
    for N in [100]:
        scores = build_kmeans_model(load_data(all_sources=False, strip_urls=True, text_only=True),
                                    build_preprocessor_pipeline(truncate=True, ncomp=N, tfidf=True, hashing=False),
                                    n_clusters=[2, 3, 4, 5])

        plt.plot([2, 3, 4, 5], scores)
        plt.xlabel('$k$')
        plt.ylabel('Silhouette Score')
        plt.savefig('kplot_n%d.png' % N)
        plt.close()
    """

    preprocessor = build_preprocessor_pipeline(truncate=True, ncomp=200, tfidf=True, hashing=False)
    data = load_data(all_sources=False, strip_urls=True, text_only=True)

    preprocessor.fit(data)
    sv = preprocessor.named_steps['svd'].singular_values_

    plt.plot(sv)
    plt.xlabel('Singular Values')
    plt.ylabel('Magnitude')
    plt.yscale('log')
    plt.savefig('singular_values.png')
    #plt.show()




