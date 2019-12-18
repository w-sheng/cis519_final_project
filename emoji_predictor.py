import math
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import pandas as pd

from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

def load_file(filename):
  '''
  Loads file of text messages containing emojis into correct format for training.
  Each line of the text messages file has the following format:
    "Name of text recipient": "text content"

  Returns: emoji_df: a dataframe with columns ['emoji','text'], where there is
           a row for each emoji every time it appears in a text
           vocab: a set of unique tokens appearing in the input texts
  '''
  RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

  # helper function to parse data from each line of file
  def parse_line(s):
    split_line = s.split(': ', 1)
    doc = split_line[0]
    text_emoji = split_line[1]

    # get text and most frequent emoji
    text = RE_EMOJI.sub('', text_emoji).strip()
    emojis = RE_EMOJI.findall(text_emoji)

    return text, emojis

  emoji_df = pd.DataFrame()
  labeled_texts = []

  with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
      m = RE_EMOJI.findall(line)
      if m:
        text, emojis = parse_line(line)
        if (text != ''):
          for e in emojis:
            labeled_texts.append([e, text])

  emoji_df = pd.DataFrame(labeled_texts)
  emoji_df.columns = ['emoji','text']

  emoji_freq_df = emoji_df.groupby('emoji').agg(['count'])
  emoji_freq_df.reset_index(inplace=True)
  emoji_freq_df.columns = ['emoji','count']
  emoji_freq_df = emoji_freq_df.sort_values(['count'], ascending=False)

  top_emojis = list(emoji_freq_df[:50]['emoji'])
  filtered_emoji_df = emoji_df.loc[emoji_df['emoji'].isin(top_emojis)]

  return filtered_emoji_df['text'].values, filtered_emoji_df['emoji'].values, emoji_freq_df['emoji'].values

def get_accuracy(y_pred, y_true):
  numerator = sum([ 1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_pred)) ])
  denominator = len(y_pred)
  return float(numerator) / float(denominator)

if __name__ == '__main__':
    texts = ["i love you",
             "oh i'm sorry i'm so lit i try to be sad",
             "I like the look of that dress",
             "lol testing this!!",
             "lol this is a test text",
             "omg i failed",
             "omg i think i failed my exam lol",
             "omg amazing thank you so much!!!!!",
             "woah this is so cool yay",
             "interesting.. this model is kinda weird"]

    X_train, y_train, emojis = load_file('../michael_emoji_train.txt')
    X_dev, y_dev, _ = load_file('../michael_emoji_test.txt')

    # get text features
    tfidf = TfidfVectorizer(max_features=10000, stop_words = list(ENGLISH_STOP_WORDS))
    tfidf.fit(X_train)
    vector = tfidf.transform(X_train)

    # choose a classifier model to train
    # clf = GaussianNB()
    clf = KNeighborsClassifier()
    # clf = MLPClassifier()
    # clf = AdaBoostClassifier()
    clf.fit(vector.todense(), y_train)

    # get accuracy score on dev set
    y_pred = []
    for text,emoji in zip(X_dev,y_dev):
      dev_tfidf = tfidf.transform([text])
      probs = clf.predict_proba(dev_tfidf.todense()).flatten()
      most_prob = np.argmax(probs)
      pred_emoji = clf.classes_[most_prob]
      y_pred.append(pred_emoji)

      # print('-->', text)
      # print('Real: ', emoji)
      # print('Pred: ', pred_emoji)
      # print('\n')

    print('Accuracy: ', get_accuracy(y_pred, y_dev))

    # predict emojis
    for text in texts:
      test_tfidf = tfidf.transform([text])
      probs = clf.predict_proba(test_tfidf.todense()).flatten()
      above_0 = np.argwhere(probs > 0).flatten()
      most_prob = np.argmax(probs)
      print('-->', text, '=', clf.classes_[most_prob])

      # other non-predicted emoji probabilities
      # print('-->', text)
      # for i in above_0:
      #   print('\t', clf.classes_[i], ': ', probs.flatten()[i])
