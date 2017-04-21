#!/usr/bin/python3
"""Migrated from CleanText.ipynb"""
import pandas
import nltk
import pickle
import collections

DATA_SIZE = 1000

PICKLE_FILE_NAME_TEMPLATE = "pickles/words%d.pickle"


def get_words(size, stem=True):
  try:
    return pickle.load(open(PICKLE_FILE_NAME_TEMPLATE % size, "rb"))
  except IOError:
    data = pandas.read_csv('data/train.csv', nrows=size)

    full_description = data["FullDescription"]

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(' '.join(full_description))
    if stem:
      # Porterstemmer lower cases words as well.
      stemmer = nltk.stem.porter.PorterStemmer()
      stemmed_tokens = []
      for token in tokens:
        try:
          stemmed_tokens.append(stemmer.stem(token))
        except IndexError:
          # Some words cause the PorterStemmer to throw an Index error. This
          # effectively skips them while printing them out.
          print(token)
      tokens = stemmed_tokens
    text = nltk.Text(tokens)
    pickle.dump(text, open(PICKLE_FILE_NAME_TEMPLATE % size, "wb"))
    return text

text = get_words(DATA_SIZE)
word_counter = collections.Counter(text).most_common()
print(word_counter)
print(len(word_counter))

# From the amazing answer on stack overflow here: http://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("words-by-frequency.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))

#infer_spaces(str.lower("PastryChefBakerartisanbakerySuffolkCoast"))
