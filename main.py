import random
import json 
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

with open("intents.json", "r") as f:
    intents = json.load(f)

words = []
classes = []
documents = []
# do it using regex
ignore = [",",".","?","!","&"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.tokenize.word_tokenize(pattern)
        words.extend(word_list)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        documents.append((word_list, intent['tag']))

# vocabulary
words = [lemma.lemmatize(word) for word in words if word not in ignore]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
opEmpty = [0]*len(classes)

for document in documents:
# Bag of words for input patterns (training sentence)
    bag = []
    wordPatterns = document[0]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    opRow = list(opEmpty)
    opRow[classes.index(document[1])] = 1
    training.append(bag + opRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(words),), activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

sgd = tf.keras.optimizers.SGD()
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5")
print("Viola!!")