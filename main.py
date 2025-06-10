# import random
import json 
import pickle
# import numpy 
# import tensorflow 

import nltk
nltk.download('wordnet')
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
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        documents.append((word_list, intent['tag']))

# vocabulary
words = [lemma.lemmatize(word_list) for word in words if word not in ignore]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wd'))
pickle.dump(classes, open('classes.pkl', 'wd'))

training = []
opEmpty = [0]*len(classes)

for document in documents:
# Bag of words for input patterns (training sentence)
    bag = []
    wordPatterns = document[0]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    opRow = list(opEmpty)
    opRow[classes.index(document[1])]
    training.append(bag + opRow)



        

