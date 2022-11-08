# References:
#   bag of words: https://www.youtube.com/watch?v=UFtXy0KRxVI
#   bag of words (referenced this implementation):
#   https://www.freecodecamp.org/news/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04/
#   stopwords: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
#   stopwords: https://pub.towardsai.net/stop-the-stopwords-using-different-python-libraries-ffa6df941653
#   regex: https://www.pythontutorial.net/python-regex/python-regex-sub/
#   regex: https://www.w3schools.com/jsref/jsref_regexp_wordchar_non.asp
#   numpy zeros: https://www.geeksforgeeks.org/numpy-zeros-python/
#
# import keras
#
# tokenize = keras.prepocessing.text.Tokenizer(num_words=400)
# tokenize.fit_on_texts(train_questions)
#
# body_train = tokenize.text_to_matrix(train_questions)
# body_test = tokenize.text_to_matrix(test_questions)
#
# model = keras.models.Sequential()
# model.add(keras.layers.Dense(50, input_shape(vocab_size,), activation='relu'))


import numpy as np  # for np.arrays and np.zeros
import re           # Support for regular expressions

# For some reason, there are troubles with NLTK importing stop words and using the built in tokenizer
from spacy.lang.en.stop_words import STOP_WORDS


def tokenize_sentence(sentences):
    words = []

    # Builds the list of words by extracting words from the sentence
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)     # appends multiple items to a list as individual items

    # Sets cannot have duplicate values, this line removes duplicates
    words = sorted(list(set(words)))

    return words


def word_extraction(sentence):
    # ignore = ['a', "the", "is"]

    # split the words list by non-word characters (spaces)
    # replaces the non-word characters with sentence as the input and " " as the replacement
    words = re.sub("\W", " ", sentence).split()

    cleaned_text = [w.lower() for w in words if w not in STOP_WORDS]
    # cleaned_text =  [w.lower() for w in words if w not in ignore]
    return cleaned_text


def generate_bag_of_words(inner_all_sentences):
    vocab = tokenize_sentence(inner_all_sentences)

    # Gives the vocabulary list of all the words we know minus the stop words
    print("Word List for entire document: ", vocab, "\n")

    # Seeing what words are at what index, can see how the sentences are converted to a "bag of words"
    all_words_with_index = []

    for i, word in enumerate(vocab):
        all_words_with_index.append((i, word))

    print(all_words_with_index, "\n")


    # loop through the sentences passed as parameters
    for sentence in inner_all_sentences:
        words = word_extraction(sentence)

        # create a bag vector of all zeros equal to the length of vocabulary
        # the length of the vector will always be equal to vocab size
        bag_vector = np.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1  # count the respective words (each word assumed an index with enumerate)

        print(sentence, np.array(bag_vector), "\n")     # print the sentence and the bag of words


all_sentences = ["Joe waited for the train", "The train was late", "Mary and Samantha took the bus",
                "I looked for Mary and Samantha at the bus station",
                "Mary and Samantha arrived at the bus station early but waited until noon for the bus"]

generate_bag_of_words(all_sentences)
