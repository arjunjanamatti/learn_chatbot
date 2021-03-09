from newspaper import Article
import random
import string
import numpy as np
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#download package from nltk
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)

#get artical url
article= Article('https://simple.wikipedia.org/wiki/Light')
article.download()
article.parse()
article.nlp()
corpus=article.text
#print
print(corpus)


#tokenization
text=corpus
sent_tokens=nltk.sent_tokenize(text)
print(sent_tokens)

#creating a dictionary to remove the punctuation
remove_punct_dict=dict( (ord(punct),None) for punct in string.punctuation)
print(string.punctuation)
print(remove_punct_dict)