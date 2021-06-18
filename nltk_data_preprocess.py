from sklearn.datasets import fetch_20newsgroups
text_data = fetch_20newsgroups()

import numpy as np

raw_text = text_data.data
raw_text_slice = raw_text[:4]

# Stage 1: Convert to lowercase

clean_text_1 = []
for words in raw_text_slice:
  clean_text_1.append(str.lower(words))
  
  
# Function for reporoducibility if needed
def toLower(data):
  clean_text_0 = []
  for word in raw_text_slice:
    clean_text_0.append(str.lower(words))
  return clean_text_0

# If needed: print(clean_text_1)

# Stage 2: tokenize

from nltk.tokenize import sent_tokenize, word_tokenize

import nltk
nltk.download('punkt')

#sentence tokenizer
sent_tok = []
for sent in clean_text_1:
  sent = sent_tokenize(sent) #convert elements of clean_text_1 into sentence tokens
  sent_tok.append(sent)

# If needed: print(sent_tok[:5])

#word tokenizer

word_tok = []
for wordi in clean_text_1:
  wordi = word_tokenize(wordi)
  word_tok.append(wordi)

# If needed: print(word_tok)

#we use regular exrpession to quickly get rid of punctuations to get only words
import re

clean_text_3 = []

for words in word_tok:
  clean = []
  for w in words:
    res = re.sub(r'[^\w\s]', "", w)
    if res != "":
      clean.append(res)
    clean_text_3.append(clean)

# If needed: print(clean_text_3[:5])

# Stage 4: StopWord removal

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

#if it is the case that word is not a stopword append to clean4
clean_text_4 = []
for words in clean_text_3:
  w = []
  for word in words:
    if not word in stopwords.words('english'):
      w.append(word)  
    clean_text_4.append(w)
# If needed: print(clean_text_4[1:2])

# We can also lemitize

from nltk.stem.wordnet import  WordNetLemmatizer
wnet = WordNetLemmatizer()

import nltk
nltk.download('wordnet')

lem = []
for words in clean_text_4:
  w = []
  for word in words:
    w.append(wnet.lemmatize(word))

  lem.append(w)
  
# If needed: print(lem[:4])


#To compare original raw text to final processed type.

# REFER TO NAIVE BAYES ALGORITH MATH EXPLANATION .md FILE #

# REFER TO NAIVE BAYES ALGORITH .py implementation FILE #

# REFER TO FULL NAIVE BAYES ALGORITH CLASSIFIER .py FILE #























