# References:
#   https://www.youtube.com/watch?v=UFtXy0KRxVI

import keras

tokenize = keras.prepocessing.text.Tokenizer(num_words=400)
tokenize.fit_on_texts(train_questions)

body_train = tokenize.text_to_matrix(train_questions)
body_test = tokenize.text_to_matrix(test_questions)