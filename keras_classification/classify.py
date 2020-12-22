import keras
import numpy as np
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def main():
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
    word_index = reuters.get_word_index(path="reuters_word_index.json")
    print('# of Training Samples: {}'.format(len(x_train)))
    print('# of Test Samples: {}'.format(len(x_test)))

    num_classes = max(y_train) + 1
    print('# of CLasses: {0}'.format(num_classes))

    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='count')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='count')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(x_train[0])
    print(len(x_train[0]))
    print(max(x_train[0]))

    print(y_train[0])
    print(len(y_train[0]))

    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    # model.add(Activation('relu'))
    model.add(Activation('exponential'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.metrics_names)

    batch_size = 32
    epochs = 2

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
