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
  RE_EMOJI = re.compile(r'\\u[0-9a-z]{4}')
  
  # helper function to parse data from each line of file
  def parse_line(s):
    split_line = s.split(': ', 1)
    doc = split_line[0]
    text_emoji = split_line[1]

    # get text and most frequent emoji
    text = RE_EMOJI.sub(r'', text_emoji)
    all_emojis = RE_EMOJI.findall(text_emoji)
    _, emoji = max([(all_emojis.count(e),e) for e in set(e)])

    return text, doc, emoji

  texts = []
  y = []
  document_names = []
  vocab_set = set{}

  with open(filename, 'rb') as f:
    for line in f:
      m = RE_EMOJI.match(line)
      if m:
        text, doc, emoji = parse_line(line)
        texts.append(text)
        document_names.append(doc)
        y.append(emoji)
        for token in text.split():
          vocab_set.add(token) 

  vocab = list(vocab_set)
  line_tuples = list(zip(document_names, [text.split() for text in texts]))

  return line_tuples, document_names, vocab, y

def load_sentence(sentence):
  vocab_set = set{}
  for token in sentence.split():
    vocab_set.add(token)
  vocab = list(vocab_set)
  
  line_tuples = [tuple(('test_doc', [sentence.split()]))]
  document_names = ['test_doc']
  
  return line_tuples, document_names, vocab

def precision(y_pred, y_true):
  numerator = sum([ 1 if y_pred[i] == 1 and y_true[i] == 1 else 0 for i in range(len(y_pred)) ])
  denominator = sum(y_pred)
  return float(numerator) / float(denominator)

def recall(y_pred, y_true):
  numerator = sum([ 1 if y_pred[i] == 1 and y_true[i] == 1 else 0 for i in range(len(y_pred)) ])
  denominator = sum(y_true)
  return float(numerator) / float(denominator)

def fscore(y_pred, y_true):
  precision = get_precision(y_pred, y_true)
  recall = get_recall(y_pred, y_true)
  return float(2 * precision * recall) / float(precision + recall)

def create_td_matrix(line_tuples, document_names, vocab):
  '''
  Inputs:
    line_tuples: list of tuples containing the name of the document and a tokenized line from that document
    document_names: list of the document names
    vocab: list of the tokens in the vocabulary

  Output:
    td_matrix: mxn numpy array where A_ij contains the frequency with which word i occurs in document j
  '''
  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  docname_to_id = dict(zip(document_names, range(0, len(document_names))))

  td_matrix = np.empty(shape=(len(vocab), len(document_names)))

  for line_tuple in line_tuples:
    document_name, tokenized_line = line_tuple
    for word in tokenized_line:
      doc_id = docname_to_id[document_name]
      vocab_id = vocab_to_id[word]
      td_matrix[vocab_id][doc_id] += 1
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
      idf = float(math.log10(N / word_count[word]))
      result[word][doc] = tf * idf

  return result

def naive_bayes(train_x, train_y, dev_x, dev_y, test_x):
  train_x_normalized = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)
  dev_x_normalized = (dev_x - dev_x.mean(axis=0)) / dev_x.std(axis=0)
  test_x_normalized = (test_x - test_x.mean(axis=0)) / test_x.std(axis=0)

  clf = GaussianNB()
  clf.fit(train_x_normalized, train_y)

  ty_pred = clf.predict(train_x_normalized)
  tprecision = precision(ty_pred, train_y)
  trecall = recall(ty_pred, train_y)
  tfscore = fscore(ty_pred, train_y)

  dy_pred = clf.predict(dev_x_normalized)
  dprecision = precision(dy_pred, dev_y)
  drecall = recall(dy_pred, dev_y)
  dfscore = fscore(dy_pred, dev_y)

  y_pred = clf.predict(test_x_normalized)

  train_performance = (tprecision, trecall, tfscore)
  dev_performance = (dprecision, drecall, dfscore)
  return train_performance, dev_performance, y_pred

if __name__ == '__main__':
    sentence_to_augment = sys.argv[1]

    line_tuples_train, document_names_train, vocab_train, train_y = load_file('data/train.txt')
    line_tuples_dev, document_names_dev, vocab_dev, dev_y = load_file('data/dev.txt')
    line_tuples_test, document_names_test, vocab_test = load_sentence(sentence_to_augment)

    td_train = create_td_matrix(line_tuples_train, document_names_train, vocab_train)
    td_dev = create_td_matrix(line_tuples_dev, document_names_dev, vocab_dev)
    td_test = create_td_matrix(line_tuples_test, document_names_test, vocab_test)

    tf_idf_train = create_tf_idf_matrix(td_train)
    tf_idf_dev = create_tf_idf_matrix(td_dev)
    tf_idf_test = create_tf_idf_matrix(td_test)

    train_performance, dev_performance, y_pred = naive_bayes(tf_idf_train, train_y, tf_idf_dev, dev_y, tf_idf_test)
    print("Training performance: ", train_performance)
    print("Development performance: ", dev_performance)
    print("Sentence to augment: ", sentence_to_augment, " | Suggested emoji: ", y_pred)
