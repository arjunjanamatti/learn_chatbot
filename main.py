import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import json
import random
import pickle

with open('intents.json') as file:
    data = json.load(file)

print(data)

try:
    with open(file='data.pickle', mode='rb') as file:
        words, labels, training, output = pickle.load(file)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            docs_x.append(tokens)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])


    # lower case all the words
    words = [stemmer.stem(word.lower()) for word in words
             if word != "?"]

    # removal of all duplicates
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(word) for word in doc]

        for word in words:
            if word in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = np.array(training)
    output = np.array(output)

    with open(file='data.pickle', mode='wb') as file:
        pickle.dump(obj=(words, labels, training, output),file=file)



net = tflearn.input_data(shape=[None, len(training[0])])

net = tflearn.fully_connected(net, n_units=8)
net = tflearn.fully_connected(net, n_units=8)
net = tflearn.fully_connected(net, n_units=len(output[0]), activation='softmax')

net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(X_inputs=training, Y_targets=output, n_epoch=1000, batch_size=8, show_metric=True)
model.save('initial_model.tflearn')