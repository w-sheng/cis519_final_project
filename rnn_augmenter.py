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
from pymagnitude import *
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class WordRNNClassify(nn.Module):
    def __init__(self, weights_matrix, input_size, output_size, vocab_size, hidden_size=1024):
        super(WordRNNClassify, self).__init__()
#       self.input_size = input_size
#       self.output_size = output_size
        self.embeddings = nn.Embedding.from_pretrained(weights_matrix)

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

def trainAll(line_tuples_train, vocab_train, train_y, word_to_ix, line_tuples_dev, dev_y, weights_matrix, test_file):
    n_categories_train = len(set(train_y))
    n_words_train = len(vocab_train)

    plot_every = 10
    n_iters = 2400
    # n_hidden = int((len(vocab_train) + len(set(train_y))) / 2)
    n_hidden = 128
    print("Hidden layer size", n_hidden)
    embedding_dim = 300

    rnn = WordRNNClassify(weights_matrix, embedding_dim, n_categories_train, n_words_train, n_hidden)

    criterion = nn.NLLLoss()
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=0.002)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.002)
    # optimizer = torch.optim.RMSprop(rnn.parameters(), lr=0.002)

    current_loss = 0
    all_losses = []
    all_accs = []
    dev_accs = []
    for iter in range(1, n_iters + 1):
        output, loss = trainOneEpoch(rnn, criterion, optimizer, line_tuples_train, train_y, word_to_ix)
        # current_loss += loss
        if iter % plot_every == 0:
            print(iter)
        #     loss = current_loss / plot_every
        #     train_acc = calculateAccuracy(rnn, line_tuples_train, train_y, word_to_ix)
        #     test_acc = calculateAccuracy(rnn, line_tuples_dev, dev_y, word_to_ix)
        #     all_losses.append(loss)
        #     all_accs.append(train_acc)
        #     dev_accs.append(test_acc)
        #     print("Iteration:", iter, "Loss", loss, "Train Acc", train_acc, "Test Acc", test_acc)
        #     current_loss = 0
            if iter == n_iters:
                test_file_open = open(test_file)
                for line in test_file_open:
                    hidden = rnn.init_hidden()
                    for word in line.split():
                        if word in word_to_ix:
                            X_train_tensor = torch.tensor([word_to_ix[word]], dtype=torch.long)
                            X_embedding = rnn.embeddings(X_train_tensor)
                            output, hidden = rnn(X_embedding, hidden)
                    top_n, top_i = output.topk(1)
                    category_i = top_i[0].item()
                    print(line, int_to_emoji[category_i], top_n[0].item())
                test_file_open.close()

    # _, axs = plt.subplots(1,3)
    # axs[0].set_title('Training Loss')
    # axs[1].set_title('Training Accuracy')
    # axs[2].set_title('Development Accuracy')
    # axs[0].plot(all_losses)
    # axs[1].plot(all_accs)
    # axs[2].plot(dev_accs)
    # axs[0].set(xlabel="Iteration (100s)")
    # axs[1].set(xlabel="Iteration (100s)")
    # axs[2].set(xlabel="Iteration (100s)")
    # axs[0].set(ylabel="Loss")
    # axs[1].set(ylabel="Accuracy")
    # axs[2].set(ylabel="Accuracy")
    # plt.show()

    return rnn

def split_file(file, file_length, sample_length, output_1, output_2):
    skipping = random.sample(range(file_length), sample_length)
    input = open(file)
    output_train = open(output_1, 'w')
    output_dev = open(output_2, 'w')
    for i, line in enumerate(input):
        if i in skipping:
            output_dev.write(line)
        else:
            output_train.write(line)
    input.close()
    output_train.close()
    output_dev.close()

if __name__ == '__main__':
    # split_file('../michael_emoji.txt', 2878, 500, '../michael_emoji_train.txt', '../michael_emoji_test.txt')

    torch.manual_seed(1)

    line_tuples_train, vocab_train, train_y = ta.load_file('../michael_emoji_train.txt')
    line_tuples_dev, vocab_dev, dev_y = ta.load_file('../michael_emoji_test.txt')

    test_vocab = set()
    test_file = open('michael-gpt2.txt')
    for line in test_file:
        for word in line.split():
            test_vocab.add(word)
    test_file.close()
    test_vocab = list(test_vocab)

    vocab = list(set(vocab_train + vocab_dev + test_vocab))
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

    vectors = Magnitude("../glove.6B.300d.magnitude")
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))

    for i, word in enumerate(vocab):
        if word in vectors:
            weights_matrix[i] = vectors.query(word)
        else:
            weights_matrix[i] = np.random.normal(scale=0.1, size=(300))

    weights_matrix = torch.FloatTensor(weights_matrix)

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

    model = trainAll(line_tuples_train, vocab, train_y, vocab_to_id, line_tuples_dev, dev_y, weights_matrix, 'michael-gpt2.txt')

    texts = ["i love you",
             "oh i'm sorry i'm so lit i try to be sad",
             "I like the look of that dress",
             "lol testing this!!",
             "lol this is a test text",
             "omg i failed",
             "omg i think i failed my exam lol",
             "omg amazing thank you so much!!!!!",
             "woah this is so cool yay",
             "interesting.. this model is kinda weird",
             "well, i'm in dave's town.",
             "no, i absolutely love paris. and i tell my friends about it when i pass by it sometimes, i even painted my bedroom windows. i hope you see me there with my friend a lot. i really are so blessed and lucky to have such a great AR",
             "i love beer",
             "oh, i'm sorry i'm so lit... i try to be funny",
             "yeah... k?",
             "its good to meet each other.",
             "well, I tried to commit suicide by squirting liquid into my heart every so often... But I just didn't die right. i'm lucky.",
             "news of her family changes your heart too.",
             "wtf, definitely made-up. it sure makes you feel weird when someone else appears.",
             "well, i'm bi, but nice guy nathan,",
             "Sure, why not from here?",
             "I will get you some sandwiches to go. Kinda what I would like to have something light.",
             "Aw, worry not we'll find something in here",
             "I like the look of that dress."]
    for text in texts:
        hidden = model.init_hidden()
        for word in text.split():
            if word in vocab_to_id:
                X_train_tensor = torch.tensor([vocab_to_id[word]], dtype=torch.long)
                X_embedding = model.embeddings(X_train_tensor)
                output, hidden = model(X_embedding, hidden)
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        print(text, int_to_emoji[category_i], top_n[0].item())
    # while True:
    #     text = str(input())
    #     if text == "kill -9":
    #         break;
    #     else:
            # hidden = model.init_hidden()
            # for word in text.split():
            #     if word in vocab_to_id:
            #         X_train_tensor = torch.tensor([vocab_to_id[word]], dtype=torch.long)
            #         X_embedding = model.embeddings(X_train_tensor)
            #         output, hidden = model(X_embedding, hidden)
            # top_n, top_i = output.topk(1)
            # category_i = top_i[0].item()
            # print(text, int_to_emoji[category_i], top_n[0].item())
            #
            # top_n, top_i = output.topk(10)
            # for i in range(10):
            #     category_i = top_i[0][i].item()
            #     print(text, int_to_emoji[category_i], top_n[0][i].item())
