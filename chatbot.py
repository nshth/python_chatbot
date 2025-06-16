"""
Chatbot Implementation File
This file handles the runtime processing of the chatbot, using a pre-trained model
to classify user input into specific intents and generate appropriate responses.
"""
import random
import json 
import numpy as np
import nltk
import re
import string

# Import configuration settings
from config import (
    INTENTS_FILE,
    WORDS_PICKLE,
    CLASSES_PICKLE,
    MODEL_FILE,
    ERROR_CUTOFF,
    EXIT_COMMANDS,
    WELCOME_MESSAGE,
    EXIT_MESSAGE,
    PROMPT_MESSAGE,
    REMOVE_STOPWORDS
)

# Convert pickle file paths to JSON file paths
words_json_file = WORDS_PICKLE.replace('.pkl', '.json')
classes_json_file = CLASSES_PICKLE.replace('.pkl', '.json')

# Download necessary NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Import stopwords for filtering
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load the pre-trained model from file
from keras.models import load_model
model = load_model(MODEL_FILE)

# Initialize lemmatizer for text processing
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intent configurations from JSON file
with open(INTENTS_FILE, 'r') as f:
    intents = json.load(f)

# Load vocabulary and intent classes using JSON instead of pickle
try:
    # Try to load from the new JSON files first
    with open(words_json_file, 'r') as f:
        words = json.load(f)
    with open(classes_json_file, 'r') as f:
        classes = json.load(f)
except FileNotFoundError:
    # Fallback to the old pickle files if JSON files don't exist yet
    import pickle
    words = pickle.load(open(WORDS_PICKLE, 'rb'))
    classes = pickle.load(open(CLASSES_PICKLE, 'rb'))
    print("Warning: Using legacy pickle files. Run main.py to generate JSON files.")

# Common contractions mapping for expansion
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mightn't": "might not",
    "might've": "might have",
    "mustn't": "must not",
    "must've": "must have",
    "needn't": "need not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

def clean_sentence(usentence):
    """
    Process user input with comprehensive text preprocessing

    Steps:
    1. Convert to lowercase
    2. Expand contractions (e.g., "don't" -> "do not")
    3. Remove punctuation and special characters
    4. Tokenize the text
    5. Remove stopwords (optional based on chatbot needs)
    6. Lemmatize words to their root form

    Args:
        usentence (str): User's input text

    Returns:
        list: List of cleaned, lemmatized words from the input
    """
    if not usentence or not isinstance(usentence, str):
        return []

    # Convert to lowercase
    usentence = usentence.lower().strip()

    # Expand contractions
    words_with_apostrophes = re.findall(r'\w+\'\w+', usentence)
    for word in words_with_apostrophes:
        if word.lower() in CONTRACTIONS:
            usentence = usentence.replace(word, CONTRACTIONS[word.lower()])

    # Remove punctuation and special characters
    # Keep apostrophes for contractions that weren't in our dictionary
    usentence = re.sub(r'[^\w\s\']', ' ', usentence)
    # Replace multiple spaces with single space
    usentence = re.sub(r'\s+', ' ', usentence).strip()

    # Tokenize
    tokens = nltk.word_tokenize(usentence)

    # Remove stopwords if configured to do so
    if REMOVE_STOPWORDS:
        tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization with part of speech tagging for better results
    lemmatized_words = []
    for word in tokens:
        # Remove any remaining punctuation
        word = word.strip(string.punctuation)
        if word:  # Skip empty strings
            lemmatized_word = lemmatizer.lemmatize(word, pos='v')  # First try as verb
            if lemmatized_word == word:  # If unchanged, try as noun
                lemmatized_word = lemmatizer.lemmatize(word, pos='n')
            lemmatized_words.append(lemmatized_word)

    return lemmatized_words

def bag_of_words(usentence):
    """
    Convert user input to bag-of-words representation

    Creates a binary vector indicating the presence of words from the
    vocabulary in the user's input

    Args:
        usentence (str): User's input text

    Returns:
        numpy.array: Binary vector representation of the input
    """
    sentence = clean_sentence(usentence)
    bag = [0]*len(words)
    for word in sentence:
        for i, w in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(usentence):
    """
    Classify user input into an intent

    Uses the neural network model to predict the most likely intent
    based on the user's input text

    Args:
        usentence (str): User's input text

    Returns:
        list: List of dictionaries containing predicted intents and their confidence scores
    """
    # Get bag-of-words representation
    bow = bag_of_words(usentence)
    # Get raw prediction from model
    prediction = model.predict(np.array([bow]))[0]
    # Use confidence threshold from config
    result = [[i, p] for i, p in enumerate(prediction) if p > ERROR_CUTOFF]
    # Sort predictions by confidence (highest first)
    result.sort(key=lambda x: x[1], reverse=True)

    # Fallback option if no intent meets the confidence threshold
    if not result:
        return [{'intent': 'fallback', 'prediction_value': '0'}]

    # Format results as list of dictionaries
    results = []
    for r in result:
        results.append({'intent': classes[r[0]], 'prediction_value': str(r[1])})

    return results


def get_answer(intent_list, intent_JSON):
    """
    Generate a response based on the predicted intent

    Args:
        intent_list (list): List of dictionaries with predicted intents
        intent_JSON (dict): The loaded intents configuration

    Returns:
        str: A randomly selected response for the predicted intent
    """
    # Get the tag of the top predicted intent
    tag = intent_list[0]['intent']
    # Find the matching intent in the configuration
    for i in intent_JSON['intents']:
        if i['tag'] == tag:
            # Randomly select one of the responses for variety
            ans = random.choice(i['responses'])
            break
    return ans

print(WELCOME_MESSAGE)
print(PROMPT_MESSAGE)

# Main chatbot interaction loop
while True:
    # Get user input
    user_msg = input("")

    # Check if user wants to exit before running prediction
    if user_msg.lower().strip() in EXIT_COMMANDS:
        print(EXIT_MESSAGE)
        break

    # Predict intent from user message
    prediction = predict_class(user_msg)
    # Generate appropriate response
    answer = get_answer(prediction, intents)
    # Display response to user
    print(answer)
