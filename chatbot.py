import random
import json 
import pickle
import numpy as np
import nltk

from keras.models import load_model
model = load_model('chatbot_model.h5')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

with open('intents.json', 'r') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_sentence(usentence):
    sentence = nltk.word_tokenize(usentence)
    sentence = [lemmatizer.lemmatize(word.lower()) for word in sentence]
    return sentence

def bag_of_words(usentence):
    sentence = clean_sentence(usentence)
    bag = [0]*len(words)
    for word in sentence:
        for i, w in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# def predict_class(usentence):
#     bow = bag_of_words(usentence)
#     # didnt get [0]
#     results.append({'intent': classes[r[0]], 'prediction_value': str(r[1])})
#     # [0.20 , 0.5 .0.67]
#     Error_cutoff = 0.25
#     result = [[i,p]for i,p in enumerate(prediction) if p > Error_cutoff] 

#     result.sort(key=lambda x: x[1], reverse = True)
#     results = []
#     for r in result:
#         results.append([{'intent': classes[r[0]], 'prediction_value': str(r[1])}])
#     return results

def predict_class(usentence):
    bow = bag_of_words(usentence)
    prediction = model.predict(np.array([bow]))[0]
    Error_cutoff = 0.25
    result = [[i, p] for i, p in enumerate(prediction) if p > Error_cutoff]
    result.sort(key=lambda x: x[1], reverse=True)

    # This line makes sure `results` always exists
    if not result:
        return [{'intent': 'fallback', 'prediction_value': '0'}]

    results = []
    for r in result:
        results.append({'intent': classes[r[0]], 'prediction_value': str(r[1])})

    return results


def get_answer(intent_list, intent_JSON):
    tag = intent_list[0]['intent']
    for i in intent_JSON['intents']:
        if i['tag'] == tag:
            ans = random.choice(i['responses'])
            break
    return ans

print("All set")

while True:
    user_msg = input("")
    prediction = predict_class(user_msg)
    answer = get_answer(prediction, intents)
    print(answer)



