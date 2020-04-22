import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from text import text
from anothertext import another_text
from termcolor import colored

import re
#
def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    long_words=[]
    # remove short word
    for oneword in newString.split():
        if len(oneword)>=3:                  
            long_words.append(oneword)
    return (" ".join(long_words)).strip()
##
## preprocess the text
#data_text=text()
#data_new = text_cleaner(data_text)
#
#def frequency(str,unique=0):
#    # break the string into list of words 
#    str_list = str.split() 
#    unique_words_ones=[]
#    # gives set of unique words 
#    unique_words = set(str_list) 
#      
#    for words in unique_words :
#        number=str_list.count(words)
#        print('Frequency of ', colored(words, 'red') , 'is :', number)
#        if int(number)==1:
#            unique_words_ones.append(number)
#    print('')
#    print('Number of unique words is: ', len(unique_words_ones)) 
#    return str_list
#allwords = frequency(data_new)
#
#
#def generate_ngrams(data_new, n):
#    # Break sentence in the token, remove empty tokens
#    tokens = [token for token in data_new.split(" ") if token != ""]
#    # Use the zip function to help us generate n-grams
#    # Concatentate the tokens into ngrams and return
#    ngrams = zip(*[tokens[i:] for i in range(n)])
#    return [" ".join(ngram) for ngram in ngrams]
#ngrams_1= generate_ngrams(data_new, n=1)
#ngrams_2= generate_ngrams(data_new, n=2)
#ngrams_3= generate_ngrams(data_new, n=3)
#
#
#ngrams_1_2_3=ngrams_1+ngrams_2+ngrams_3
#
## creataing dictionaries with frequencies
#from collections import defaultdict
#ngrams_1dict= defaultdict( int )
#for item in ngrams_1:
#    ngrams_1dict[item] += 1
#    
#ngrams_2dict= defaultdict( int )
#for item in ngrams_2:
#    ngrams_2dict[item] += 1
#
#ngrams_3dict= defaultdict( int )
#for item in ngrams_3:
#    ngrams_3dict[item] += 1
#    
#ngrams_dict={}   
##ngrams_dict.update(ngrams_1dict)
##ngrams_dict.update(ngrams_3dict)
#ngrams_dict.update(ngrams_2dict)
#
## Now use those frequencies to generate
## language: from the unigram, bigram, and trigram models, in turn, generate a
## 100- word text by making random choices according to the frequency counts.
#
#import random    
#def synthesize_sentence(ngrams_dict,sentence='',f=5):
#    print ("generating random sentence:")
#    length=len(sentence)
#    if length==0:
#        sentence=random.choice(list(ngrams_dict.keys()))
#    last_word=sentence 
#    for x in range(0,100):
#        if last_word in ngrams_dict.keys():
#            next=random.choice(list(ngrams_dict.keys()))
#        else:
#            if ngrams_dict.values()>f:
#                next=random.choice(list(ngrams_dict.keys()))        
#        sentence=sentence+' '+next
#        length+=len(next.split())    
#        if length>=100:
#            return sentence+'.'
#    return sentence+'.'
#sentence = synthesize_sentence(ngrams_dict, sentence='',f=5)
#print(sentence)
#print("")
#print("")

###########################################################################
##################we train a language model################################
###########################################################################
import string
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens
 
# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
# load document
#in_filename = 'text1.txt'
#doc = load_doc(in_filename)
## clean document
#tokens = clean_doc(doc)
## organize into sequences of tokens
#length = 2
#sequences = list()
#for i in range(length, len(tokens)):
#	# select sequence of tokens
#	seq = tokens[i-length:i]
#	# convert into a line
#	line = ' '.join(seq)
#	# store
#	sequences.append(line)
## save sequences to file
#out_filename = 'text2_1_sequences.txt'
#save_doc(sequences, out_filename)
## load sequences
#in_filename = 'text2_1_sequences.txt'
#doc = load_doc(in_filename)
#lines = doc.split('\n')
#import keras
## integer encode sequences of words
#tokenizer = keras.preprocessing.text.Tokenizer()
#tokenizer.fit_on_texts(lines)
#sequences = tokenizer.texts_to_sequences(lines)
## vocabulary size
#vocab_size = len(tokenizer.word_index) + 1
## separate into input and output
#sequences = np.array(sequences)
#X, y = sequences[:,:-1], sequences[:,-1]
#y = to_categorical(y, num_classes=vocab_size)
#seq_length = X.shape[1]
## define model
#model = Sequential()
#model.add(Embedding(vocab_size, 50, input_length=seq_length))
#model.add(LSTM(8, return_sequences=True))
#model.add(LSTM(8))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(vocab_size, activation='softmax'))
#print(model.summary())
## compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
## fit model
#model.fit(X, y, batch_size=64, epochs=10)
#
#
#
#
#
#
#
##
##
### load document
##in_filename = 'text2.txt'
##doc = load_doc(in_filename)
### clean document
##tokens = clean_doc(doc)
### organize into sequences of tokens
##length = 2
##sequences = list()
##for i in range(length, len(tokens)):
##	# select sequence of tokens
##	seq = tokens[i-length:i]
##	# convert into a line
##	line = ' '.join(seq)
##	# store
##	sequences.append(line)
### save sequences to file
##out_filename = 'text2_2_sequences.txt'
##save_doc(sequences, out_filename)
### load sequences
##in_filename = 'text2_2_sequences.txt'
##doc = load_doc(in_filename)
##lines = doc.split('\n')
##import keras
### integer encode sequences of words
##tokenizer = keras.preprocessing.text.Tokenizer()
##tokenizer.fit_on_texts(lines)
##sequences = tokenizer.texts_to_sequences(lines)
### vocabulary size
##vocab_size = len(tokenizer.word_index) + 1
### separate into input and output
##sequences = np.array(sequences)
##X, y = sequences[:,:-1], sequences[:,-1]
##y = to_categorical(y, num_classes=vocab_size)
##seq_length = X.shape[1]
### define model
##model = Sequential()
##model.add(Embedding(vocab_size, 50, input_length=seq_length))
##model.add(LSTM(8, return_sequences=True))
##model.add(LSTM(8))
##model.add(Dense(8, activation='relu'))
##model.add(Dense(vocab_size, activation='softmax'))
##print(model.summary())
### compile model
##model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
### fit model
##model.fit(X, y, batch_size=64, epochs=10)
#
#
#
#
#
#
#
##
### load document
#in_filename = 'text1.txt'
#doc = load_doc(in_filename)
## clean document
#tokens = clean_doc(doc)
## organize into sequences of tokens
#length = 3
#sequences = list()
#for i in range(length, len(tokens)):
#	# select sequence of tokens
#	seq = tokens[i-length:i]
#	# convert into a line
#	line = ' '.join(seq)
#	# store
#	sequences.append(line)
## save sequences to file
#out_filename = 'text3_1_sequences.txt'
#save_doc(sequences, out_filename)
## load sequences
#in_filename = 'text3_1_sequences.txt'
#doc = load_doc(in_filename)
#lines = doc.split('\n')
#import keras
## integer encode sequences of words
#tokenizer = keras.preprocessing.text.Tokenizer()
#tokenizer.fit_on_texts(lines)
#sequences = tokenizer.texts_to_sequences(lines)
## vocabulary size
#vocab_size = len(tokenizer.word_index) + 1
## separate into input and output
#sequences = np.array(sequences)
#X, y = sequences[:,:-1], sequences[:,-1]
#y = to_categorical(y, num_classes=vocab_size)
#seq_length = X.shape[1]
## define model
#model = Sequential()
#model.add(Embedding(vocab_size, 50, input_length=seq_length))
#model.add(LSTM(8, return_sequences=True))
#model.add(LSTM(8))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(vocab_size, activation='softmax'))
#print(model.summary())
## compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
## fit model
#model.fit(X, y, batch_size=64, epochs=10)
##
##
##
##
###
###
###
###
###
#### load document
#in_filename = 'text2.txt'
#doc = load_doc(in_filename)
## clean document
#tokens = clean_doc(doc)
## organize into sequences of tokens
#length = 3
#sequences = list()
#for i in range(length, len(tokens)):
#	# select sequence of tokens
#	seq = tokens[i-length:i]
#	# convert into a line
#	line = ' '.join(seq)
#	# store
#	sequences.append(line)
## save sequences to file
#out_filename = 'text3_2_sequences.txt'
#save_doc(sequences, out_filename)
## load sequences
#in_filename = 'text3_2_sequences.txt'
#doc = load_doc(in_filename)
#lines = doc.split('\n')
#import keras
## integer encode sequences of words
#tokenizer = keras.preprocessing.text.Tokenizer()
#tokenizer.fit_on_texts(lines)
#sequences = tokenizer.texts_to_sequences(lines)
## vocabulary size
#vocab_size = len(tokenizer.word_index) + 1
## separate into input and output
#sequences = np.array(sequences)
#X, y = sequences[:,:-1], sequences[:,-1]
#y = to_categorical(y, num_classes=vocab_size)
#seq_length = X.shape[1]
## define model
#model = Sequential()
#model.add(Embedding(vocab_size, 50, input_length=seq_length))
#model.add(LSTM(8, return_sequences=True))
#model.add(LSTM(8))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(vocab_size, activation='softmax'))
#print(model.summary())
## compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
## fit model
#model.fit(X, y, batch_size=64, epochs=10)
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
## load document
in_filename = 'popular_scientific.txt'
doc = load_doc(in_filename)
# clean document
tokens = clean_doc(doc)
# organize into sequences of tokens
length = 3
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
# save sequences to file
out_filename = 'popular_scientific3_2_sequences.txt'
save_doc(sequences, out_filename)
# load sequences
in_filename = 'popular_scientific3_2_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
import keras
# integer encode sequences of words
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
# separate into input and output
sequences = np.array(sequences)

X_test, y_test = sequences[:,:-1], sequences[:,-1]
y_test = to_categorical(y_test, num_classes=vocab_size)
seq_length = X_test.shape[1]


import tensorflow


def categorical_accuracy(y_true, y_pred):
    return keras.cast(keras.equal(keras.argmax(y_test, axis=-1), keras.argmax(y_pred, axis=-1)), keras.floatx())

PREDICTED_CLASSES=model.predict(X_test, batch_size=64)
s=keras.metrics.categorical_accuracy(y_test, PREDICTED_CLASSES)
temp = sum(m)
temp/len(y_test)