import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import re

# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    P = len(series)
    X = [series[i:i+window_size] for i in range(P-window_size)]
    y = [series[i+window_size] for i in range(P-window_size)]
    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    """
    layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    layer 2 uses a fully connected module with one unit
    the 'mean_squared_error' loss should be used (remember: we are performing regression here)
    """
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # now model.output_shape == (None, 5)
    # note: `None` is the batch dimension.

    # fully connected layer with one unit
    model.add(Dense(1))

    # compile
    """
    model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy']))
    """
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    """
    text = text.replace('&',' ')
    text = text.replace('-',' ')
    text = text.replace('è',' ')
    text = text.replace('à',' ')
    text = text.replace('é',' ')
    text = text.replace('(',' ')
    text = text.replace(')',' ')
    text = text.replace(':',' ')
    text = text.replace(';',' ')
    text = text.replace('%',' ')
    text = text.replace('/',' ')
    text = text.replace('$',' ')
    text = text.replace('â',' ')
    text = text.replace('*',' ')
    text = text.replace('@',' ')
    """
    return re.sub(r'[^a-z!,.:;?]', ' ', text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    P = len(text)
    inputs = [text[i:i+window_size] for i in range(0, P-window_size, step_size)]
    outputs = [text[i+window_size] for i in range(0, P-window_size, step_size)]

    # reshape each
    #inputs = np.asarray(inputs)
    #inputs.shape = (np.shape(inputs)[0:2])
    #outputs = np.asarray(outputs)
    #outputs.shape = (len(outputs),1)

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    """
    layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    layer 3 should be a softmax activation ( since we are solving a multiclass classification)
    Use the categorical_crossentropy loss
    """
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # now model.output_shape == (None, 5)
    # note: `None` is the batch dimension.

    # fully connected layer with one unit
    model.add(Dense(num_chars, activation='softmax'))

    return model
