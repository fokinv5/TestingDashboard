import os
from optparse import OptionParser
import re
import regex
import sqlite3
import defusedxml
from defusedxml.ElementTree import fromstring as safe_fromstring
import pandas as pd
import requests
import streamlit as st
from nltk import FreqDist, WordNetLemmatizer, pos_tag
from nltk.metrics.aline import similarity_matrix
from nltk.translate import *
from nltk.translate.meteor_score import meteor_score as meteor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit import components
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import pyLDAvis.gensim
import pickle
import pyLDAvis
import numpy as np
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from pprint import pprint
from pathlib import Path
import glob
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly"

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

URL = 'https://dev.to/api/articles'
params = {
    'tag' : 'testing',
    'page' : '>1',
    'per_page' : '1000'
}

response = requests.get(url=URL, params=params)
response.raise_for_status()
articles = response.json()

df = pd.DataFrame(articles)

df.to_csv('articles.csv', index=False)
articles = pd.read_csv('articles.csv')

articles = articles['description']

articles = \
articles.map(lambda x: re.sub('[,.!?]', '', x))

articles = \
articles.map(lambda x: x.lower())

lemmatizer = WordNetLemmatizer()

stop_plus = ['start' ,'constantly','free', 'without', 'write', 'work', 'future', 'career', 'evidence', 'essential' ,'page','last' ,'week', 'weeks' ,'guide', 'quality','agents' ,'⭐️⭐️' ,'build', 'built', 'building', 'coding' ,'code', '2026', 'software', 'test', 'tests', 'testing', 'every', 'minutes', 'time', 'one', 'two', 'three', 'see', 'met', 'part', 'possible', 'still', 'way', 'says', 'keep', 'tldr', 'first', 'using', 'actually', 'answer', 'bangalore', 'often', 'move', 'real', 'across', 'small', 'new', 'made', 'team', 'may', 'like', 'whether', 'someone', 'question', 'buy', 'looks', 'look', 'need', 'something', 'know', 'telegram', 'use', 'using', 'never', 'nothing', 'right', 'thing', 'dr', 'tl', 'open', 'five', 'strength', 'hundreds', 'get', 'best', 'post', 'elevate', 'checks', 'change', 'problem' ]
stop_words = stopwords.words('english')
stop_words.extend(stop_plus)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

data = articles.tolist()
data_words = list(sent_to_words(data))

articles_lem = []

blog_data = ','.join(articles.astype(str))
tokenised = word_tokenize(blog_data)
tagged_tokens = pos_tag(tokenised)

for word, tag in tagged_tokens:
        articles_lem.append(
            lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
data_words = word_tokenize(' '.join(articles_lem))


data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

id2word = corpora.Dictionary(data_words)
texts = data_words
corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[:1][0][:30])

if __name__ == '__main__':
   num_topics = 5
   lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
   print(lda_model.print_topics())
   doc_lda = lda_model[corpus]

   if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, R=15)
        html_string = pyLDAvis.prepared_data_to_html(LDAvis_prepared)
        pyLDAvis.save_html(LDAvis_prepared, 'blogs_lda.html')