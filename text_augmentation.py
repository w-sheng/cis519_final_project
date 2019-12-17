import math
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

from collections import defaultdict
from sklearn.naive_bayes import GaussianNB

def load_file(filename):
  '''
  Loads file of text messages containing emojis into correct format for training.
  Each line of the text messages file has the following format:
    "Name of text recipient": "text content"

  Returns: line_tuples: tuple of document name and list of tokens in each text
           document_names: list of names of text recipient of a text
           vocab: list of unique tokens in all text messages
           y: list of most frequent emoji in each text message
  '''
  RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
  # helper function to parse data from each line of file
  def parse_line(s):
    split_line = s.split(': ', 1)
    doc = split_line[0]
    text_emoji = split_line[1]

    # get text and most frequent emoji
    text = RE_EMOJI.sub('', text_emoji).strip()
    all_emojis = RE_EMOJI.findall(text_emoji)
    _, emoji = max([(all_emojis.count(e),e) for e in set(all_emojis)])

    return text, emoji

  texts = []
  y = []
  vocab_set = set()

  with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
      m = RE_EMOJI.findall(line)
      if m:
        text, emoji = parse_line(line)
        if (text != ''):
          texts.append(text)
          y.append(emoji)
          for token in text.split():
            vocab_set.add(token)

  vocab = list(vocab_set)
  line_tuples = [text.split() for text in texts]

  return line_tuples, vocab, y

def load_sentence(sentence):
  vocab_set = set()
  for token in sentence.split():
    vocab_set.add(token)
  vocab = list(vocab_set)

  line_tuples = [sentence.split()]

  return line_tuples, vocab

def precision(y_pred, y_true):
  numerator = sum([ 1 if y_pred[i] == 1 and y_true[i] == 1 else 0 for i in range(len(y_pred)) ])
  denominator = sum(y_pred)
  return float(numerator) / float(denominator)

def recall(y_pred, y_true):
  numerator = sum([ 1 if y_pred[i] == 1 and y_true[i] == 1 else 0 for i in range(len(y_pred)) ])
  denominator = sum(y_true)
  return float(numerator) / float(denominator)

def fscore(y_pred, y_true):
  prec = precision(y_pred, y_true)
  rec = recall(y_pred, y_true)
  return float(2 * prec * rec) / float(prec + rec)

def create_td_matrix(line_tuples, vocab):
  '''
  Inputs:
    line_tuples: list of tuples containing the name of the document and a tokenized line from that document
    document_names: list of the document names
    vocab: list of the tokens in the vocabulary

  Output:
    td_matrix: mxn numpy array where A_ij contains the frequency with which word i occurs in document j
  '''
  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  # docname_to_id = dict(zip(document_names, range(0, len(document_names))))

  td_matrix = np.empty(shape=(len(vocab), len(line_tuples)))

  for i,line in enumerate(line_tuples):
    for word in line:
      vocab_id = vocab_to_id[word]
      td_matrix[vocab_id][i] += 1
  return td_matrix

def create_tf_idf_matrix(term_document_matrix):
  N = len(term_document_matrix[0])

  result = np.empty(shape=(len(term_document_matrix), len(term_document_matrix[0])))

  word_count = {}
  for word in range(0, len(term_document_matrix)):
    count = 0
    for doc in range(0, N):
      if term_document_matrix[word][doc] > 0:
        count += 1
    word_count[word] = count

  for word in range(0, len(term_document_matrix)):
    for doc in range(0, N):
      count = term_document_matrix[word][doc]
      tf = float(1 + math.log10(count)) if count > 0 else 0
      idf = float(math.log10(N / word_count[word])) if word_count[word] > 0 else 0
      result[word][doc] = tf * idf

  return result

def match_features(features_1, features_2, vocab_1, vocab_2):
  result = np.zeros(features_1.shape)
  print(result.shape)
  for i in range(len(vocab_1)):
    if vocab_1[i] in vocab_2:
      index = np.where(vocab_2 == vocab_1[i])
      print(index)
      result[i] = features_2[index[0]]
  return result

def naive_bayes(train_x, train_y):
  train_x_std = np.array(train_x.std(axis=0))
  # dev_x_std = np.array(dev_x.std(axis=0))
  # test_x_std = np.array(test_x.std(axis=0))
  train_x_normalized = np.array(train_x - train_x.mean(axis=0))
  # dev_x_normalized = np.array(dev_x - dev_x.mean(axis=0))
  # test_x_normalized = np.array(test_x - test_x.mean(axis=0))
  train_x_normalized = np.divide(train_x_normalized, train_x_std, out=np.zeros_like(train_x_normalized), where=train_x_std!=0)
  # dev_x_normalized = np.divide(dev_x_normalized, dev_x_std, out=np.zeros_like(dev_x_normalized), where=dev_x_std!=0)
  # test_x_normalized = np.divide(test_x_normalized, test_x_std, out=np.zeros_like(test_x_normalized), where=test_x_std!=0)
  train_x_normalized = np.transpose(train_x_normalized)
  # dev_x_normalized = np.transpose(dev_x_normalized)
  # test_x_normalized = np.transpose(test_x_normalized)

  train_y_np = np.array(train_y)
  # dev_y_np = np.array(dev_y)

  clf = GaussianNB()
  clf.fit(train_x_normalized, train_y_np)

  # print(train_x_normalized.shape, train_y_np.shape)
  # ty_pred = clf.predict(train_x_normalized)
  # # tprecision = precision(ty_pred, train_y_np)
  # # trecall = recall(ty_pred, train_y_np)
  # # tfscore = fscore(ty_pred, train_y_np)

  # print(dev_x_normalized.shape, dev_y_np.shape)
  # dy_pred = clf.predict(dev_x_normalized)
  # # dprecision = precision(dy_pred, dev_y_np)
  # # drecall = recall(dy_pred, dev_y_np)
  # # dfscore = fscore(dy_pred, dev_y_np)

  # print(test_x_normalized.shape)
  # y_pred = clf.predict(test_x_normalized)

  # train_performance = (tprecision, trecall, tfscore)
  # dev_performance = (dprecision, drecall, dfscore)
  return clf

if __name__ == '__main__':
    texts = ["i love you", "oh i'm sorry i'm so lit i try to be funny", "I like the look of that dress"]

    line_tuples_train, vocab_train, train_y = load_file('emoji_file_train.txt')
    line_tuples_dev, vocab_dev, dev_y = load_file('emoji_file_dev.txt')

    emoji_to_int = dict()
    int_to_emoji = dict()
    emoji_to_freq = dict()

    list_of_emojis = []
    new_y = []
    counter = 0
    for emoji in train_y:
      if (emoji in emoji_to_int):
        new_y.append(emoji_to_int[emoji])
        emoji_to_freq[emoji] += 1
      else:
        new_y.append(counter)
        emoji_to_int[emoji] = counter
        int_to_emoji[counter] = emoji
        emoji_to_freq[emoji] = 1
        counter += 1
    train_y = new_y

    # sorted_freq = {k: v for k, v in sorted(emoji_to_freq.items(), key=lambda item: item[1], reverse=True)}
    # for e,f in sorted_freq.items():
    #   print(e, ': ', str(f))

    new_y = []
    for emoji in dev_y:
      if emoji in emoji_to_int:
        new_y.append(emoji_to_int[emoji])
    dev_y = new_y

    vocab = set(vocab_train + vocab_dev)

    td_train = create_td_matrix(line_tuples_train, vocab)
    # td_dev = create_td_matrix(line_tuples_dev, vocab)

    tf_idf_train = create_tf_idf_matrix(td_train)
    # tf_idf_dev = create_tf_idf_matrix(td_dev)

    print(tf_idf_train.shape)
    # print(tf_idf_dev.shape)

    # matched_train = match_features(tf_idf_train, tf_idf_dev, vocab_train, vocab_dev)
    # matched_test = match_features(tf_idf_train, tf_idf_test, vocab_train, vocab_test)
    clf = naive_bayes(tf_idf_train, train_y)

    for t in texts:
      print(t)
      line_tuples_test, vocab_test = load_sentence(t)
      td_test = create_td_matrix(line_tuples_test, vocab)
      test_x = create_tf_idf_matrix(td_test)

      test_x_std = np.array(test_x.std(axis=0))
      test_x_normalized = np.array(test_x - test_x.mean(axis=0))
      test_x_normalized = np.divide(test_x_normalized, test_x_std, out=np.zeros_like(test_x_normalized), where=test_x_std!=0)
      test_x_normalized = np.transpose(test_x_normalized)

      y_pred = clf.predict(test_x_normalized)

      print("Sentence to augment: ", t, " | Suggested emoji: ", int_to_emoji[y_pred[0]])
