#! /usr/bin/python3

from sklearn.externals import joblib

import numpy as np
import pandas
import preprocess_text_data
import sklearn.decomposition

INPUT_FILE = 'train.csv'
OUTPUT_FILE = '%s_lda_30.csv' % INPUT_FILE.split(".")[0]
MIN_WORD_COUNT = 50

X, vocab = preprocess_text_data.get_lda_training_data(
    INPUT_FILE, 'CompleteJobListing', MIN_WORD_COUNT)

model = sklearn.decomposition.LatentDirichletAllocation(
    n_topics=30, max_iter=1000, random_state=1, learning_method='online',
    batch_size=16384, n_jobs=-1, verbose=10)

model.fit_transform(X)

joblib.dump(model, 'lda_model.pkl')

model = joblib.load('lda_model.pkl')

topic_word = model.components_
n_top_words = 15

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# This is, our 30 dimensional representation of the original data.

df = pandas.DataFrame(topic_word)
df.to_csv('data/%s' % OUTPUT_FILE, index=False)
