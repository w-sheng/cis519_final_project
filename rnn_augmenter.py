import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import math
import random
import string
import time
import numpy as np
import text_augmentation as ta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class WordRNNClassify(nn.Module):
    def __init__(self, input_size, output_size, vocab_size, hidden_size=128):
        super(WordRNNClassify, self).__init__()
#       self.input_size = input_size
#       self.output_size = output_size
        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
#       self.hidden = self.init_hidden()

    def forward(self, input, hidden=None):
        h, _ = hidden
        combined = torch.cat((input, h), 1)
        hidden = self.lstm(input, hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)

'''
Input: trained model, list of texts, list of class labels as integers
Output: accuracy of given model on given input X and target y
'''
def calculateAccuracy(model, X, y, word_to_ix):
    y_pred = []
    for i in range(len(X)):
        hidden = model.init_hidden()
        for word in X[i]:
            X_train_tensor = torch.tensor([word_to_ix[word]], dtype=torch.long)
            X_embedding = model.embeddings(X_train_tensor)
            output, hidden = model(X_embedding, hidden)
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        y_pred.append(category_i)
    return accuracy_score(y_pred, y)

'''
Train the model for one epoch/training text
Input: X and y are lists of tokens and classes as integers respectively
'''
def trainOneEpoch(model, criterion, optimizer, X, y, word_to_ix):
    hidden = model.init_hidden()
    model.zero_grad()

    inx = int(random.choice(range(len(X))))
    while len(X[inx]) == 0:
        inx = int(random.choice(range(len(X))))

    y_train_tensor = Variable(torch.tensor([y[inx]]))
    for word in X[inx]:
        X_train_tensor = torch.tensor([word_to_ix[word]], dtype=torch.long)
        X_embedding = model.embeddings(X_train_tensor)
        output, hidden = model(X_embedding, hidden)

    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

def trainAll(line_tuples_train, vocab_train, train_y, word_to_ix, line_tuples_dev, dev_y):
    n_categories_train = len(set(train_y))
    n_words_train = len(vocab_train)

    plot_every = 1000
    n_iters = 20000
    n_hidden = 128
    embedding_dim = 100

    rnn = WordRNNClassify(embedding_dim, n_categories_train, n_words_train, n_hidden)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.002)

    current_loss = 0
    all_losses = []
    all_accs = []
    dev_accs = []
    for iter in range(1, n_iters + 1):
        output, loss = trainOneEpoch(rnn, criterion, optimizer, line_tuples_train, train_y, word_to_ix)
        current_loss += loss
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            all_accs.append(calculateAccuracy(rnn, line_tuples_train, train_y, word_to_ix))
            dev_accs.append(calculateAccuracy(rnn, line_tuples_dev, dev_y, word_to_ix))
            current_loss = 0
            print(iter)

    _, axs = plt.subplots(1,3)
    axs[0].set_title('Training Loss')
    axs[1].set_title('Training Accuracy')
    axs[2].set_title('Development Accuracy')
    axs[0].plot(all_losses)
    axs[1].plot(all_accs)
    axs[2].plot(dev_accs)
    axs[0].set(xlabel="Iteration")
    axs[1].set(xlabel="Iteration")
    axs[2].set(xlabel="Iteration")
    axs[0].set(ylabel="Loss")
    axs[1].set(ylabel="Accuracy")
    axs[2].set(ylabel="Accuracy")
    plt.show()

    return rnn

if __name__ == '__main__':
    torch.manual_seed(1)
    line_tuples_train, vocab_train, train_y = ta.load_file('../emoji_file.txt')
    line_tuples_dev, vocab_dev, dev_y = ta.load_file('../test_file.txt')
    vocab = list(set(vocab_train + vocab_dev))
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

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

    new_y = []
    for emoji in dev_y:
        if emoji in emoji_to_int:
            new_y.append(emoji_to_int[emoji])
        else:
            new_y.append(-1)
    dev_y = new_y

    model = trainAll(line_tuples_train, vocab, train_y, vocab_to_id, line_tuples_dev, dev_y)

    texts = ["Haha funny", "nice", "wow", "damn", "holy"]
    hidden = model.init_hidden()
    for text in texts:
        for word in text.split():
            if word in vocab_to_id:
                X_train_tensor = torch.tensor([vocab_to_id[word]], dtype=torch.long)
                X_embedding = model.embeddings(X_train_tensor)
                output, hidden = model(X_embedding, hidden)
        top_n, top_i = output.topk(10)
        for i in range(10):
            category_i = top_i[0][i].item()
            print(text, int_to_emoji[category_i], top_n[0][i].item())
