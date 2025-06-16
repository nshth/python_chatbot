import random
import json 
import numpy as np
import tensorflow as tf

# Import configuration settings
from config import (
    INTENTS_FILE,
    WORDS_PICKLE,
    CLASSES_PICKLE,
    MODEL_FILE,
    EPOCHS,
    BATCH_SIZE,
    DROPOUT_RATE,
    LEARNING_RATE,
    HIDDEN_LAYER_SIZES,
    IGNORE_CHARS
)

# Import necessary NLTK libraries
import nltk
# Download required NLTK datasets
nltk.download('wordnet')  # For lemmatization
nltk.download('punkt')  # For tokenization
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer to get the root form of words
lemma = WordNetLemmatizer()

# Load the intents data from JSON file
with open(INTENTS_FILE, "r") as f:
    intents = json.load(f)

# Initialize empty lists for preprocessing
words = []  # All unique words in the patterns
classes = []  # All unique intent tags
documents = []  # Combination of patterns and their associated tags

# Process each intent and its patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern into individual words
        word_list = nltk.tokenize.word_tokenize(pattern)
        # Add words to the words list
        words.extend(word_list)
        # Add intent tag to classes if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        # Add the word list and its associated tag to documents
        documents.append((word_list, intent['tag']))

# Lemmatize all words and remove ignored characters
words = [lemma.lemmatize(word) for word in words if word not in IGNORE_CHARS]
# Remove duplicates and sort the words
words = sorted(set(words))

# Sort the classes (intent tags)
classes = sorted(set(classes))

# Save the words and classes lists using JSON instead of pickle
# Replace .pkl extensions with .json in the filenames
words_json_file = WORDS_PICKLE.replace('.pkl', '.json')
classes_json_file = CLASSES_PICKLE.replace('.pkl', '.json')

with open(words_json_file, 'w') as f:
    json.dump(words, f)

with open(classes_json_file, 'w') as f:
    json.dump(classes, f)

# Prepare training data
training = []
# Create an empty output row with zeros (one zero for each class)
opEmpty = [0]*len(classes)

# Process each document (pattern-tag pair)
for document in documents:
    # Initialize bag of words for current pattern
    bag = []
    # Get the tokenized words from the document
    wordPatterns = document[0]
    # Create the bag of words representation
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    # Create output row for current document
    opRow = list(opEmpty)
    # Set 1 at the index corresponding to the current tag
    opRow[classes.index(document[1])] = 1
    # Add the bag of words and the output row to training data
    training.append(bag + opRow)

# Shuffle the training data to prevent overfitting
random.shuffle(training)
training = np.array(training)

# Split features (bag of words) and target values (intent tags)
trainX = training[:, :len(words)]  # Bag of words features
trainY = training[:, len(words):]  # Intent tags (one-hot encoded)

# Build the neural network model
model = tf.keras.Sequential()
# First dense layer with ReLU activation
model.add(tf.keras.layers.Dense(HIDDEN_LAYER_SIZES[0], input_shape=(len(words),), activation="relu"))
model.add(tf.keras.layers.Dropout(DROPOUT_RATE))  # Add dropout to prevent overfitting
# Second dense layer
model.add(tf.keras.layers.Dense(HIDDEN_LAYER_SIZES[1], activation="relu"))
model.add(tf.keras.layers.Dropout(DROPOUT_RATE))  # Add dropout to prevent overfitting
# Output layer with softmax activation for multi-class classification
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

# Compile the model with Stochastic Gradient Descent optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(np.array(trainX), np.array(trainY), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Save the trained model to a file
model.save(MODEL_FILE)
print("Viola!!")  # Training complete
