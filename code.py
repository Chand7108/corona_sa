# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:39:51 2020

@author: Chand
"""

import pandas as pd
import numpy as np

# import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords

# import scattertext as st

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from collections import Counter

data = pd.read_csv("data.CSV")
text_data = data.text

text_data = text_data.str.replace('\n', ' ').str.replace('#', '') #Remove hashtags and newline characters
text_data = text_data.str.replace('@\w*', '') #Remove account tags
text_data = text_data.str.replace('https://[^/]*/[\w]*', '') #Remove links
text_data = text_data.str.replace('(?<=[a-z])(?=[A-Z])', ' ') #Camelcase to different words (sentence case)
text_data = text_data.str.replace('[ ]+', ' ') #Remove multiple spaces
text_data = text_data.str.replace('[.,\/#!$%\^&\*;:{}=\-_`~()]', '') #Remove punctuations
text_data = text_data.str.lower().drop_duplicates()
                                    
wtsp_tkniser = WhitespaceTokenizer()
lemmatiser = WordNetLemmatizer()
sw = set(stopwords.words('english'))
nltk.download('words')
english_words = set(nltk.corpus.words.words())

def lemmatise_text(text):
    l = []
    for w in wtsp_tkniser.tokenize(text):
        if len(w) > 2 and (not w in sw and w.lower() in english_words) or not w.isalpha():
            l.append(lemmatiser.lemmatize(w))
    return l

nltk.download('wordnet')
text_data = text_data.apply(lemmatise_text)

data['parsed_text'] = text_data

text_for_wc = [' '.join(x) for x in text_data]
text_for_wc = ' '.join(text_for_wc[:10000]) #I have taken only 1000 tweets here coz WordCloud takes a lot of time

wc = WordCloud().generate(text_for_wc)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wc)
plt.axis("off")

plt.show()

data = data.dropna(subset=['parsed_text'])

def list_to_str(l):
    return ' '.join(l)

text_data_str = text_data.apply(list_to_str)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks.callbacks import ModelCheckpoint

test_data = text_data_str

train_data = pd.read_csv('train.tsv', sep='\t')

train_data[train_data['label']==0].size

test_data = test_data.dropna()

a = test_data.values
b = train_data['sentence'].values

max_features=200

tokenizer = Tokenizer(num_words=max_features, split=' ')

tokenizer.fit_on_texts(np.append(a, b))

X = tokenizer.texts_to_sequences(train_data['sentence'].values)
X = pad_sequences(X, maxlen=37)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

print(model.summary())

Y = pd.get_dummies(train_data['label']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)

batch_size = 32
model.fit(X, Y, validation_split=0.3, epochs = 2, batch_size=batch_size, verbose = 1, callbacks=[checkpointer])

model.evaluate(X_test, Y_test)

X_pred = tokenizer.texts_to_sequences(test_data.values)
X_pred = pad_sequences(X_pred, maxlen=37)

result = model.predict(X_pred, verbose=1)
result = pd.DataFrame(result)

result.to_csv('result1.csv')