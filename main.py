from __future__ import absolute_import

import os
from optparse import OptionParser
import re
import regex
import sqlite3
import pandas as pd
import requests
import streamlit as st
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

import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from pprint import pprint
#

# nltk.download('punkt')
# nltk.download('punkt_tab')

st.title("software testing nation make some noise!!!!")
st.header("Software testing blogosphere hot topics")

#api
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
#st.dataframe(df)

df.to_csv('articles.csv', index=False)
articles = pd.read_csv('articles.csv')

# text_data = " ".join(df['description'].astype(str))
# text_data = re.sub(r'[^A-Za-z\s]', '', text_data)
# text_data = text_data.lower()
#st_words = set(STOPWORDS)
#more_stopwords = {'test', 'tests', 'testing', 'every', 'minutes', 'time', 'one', 'two', 'three', 'see', 'met', 'part', 'possible', 'still', 'way', 'says', 'keep', 'tldr', 'first', 'using' }
#st_words = st_words.union(more_stopwords)
#text_data = ' '.join(word for word in text_data.split() if word not in st_words)

#articles = articles.drop(columns=['type_of', 'id', 'readable_publish_date', 'slug', 'path', 'url', 'comments', 'public_reactions_count', 'collection_id', 'published_timestamp', 'language', 'subforem_id', 'positive_reactions_count', 'cover_image', 'social_image', 'canonical_url', 'created_at', 'edited_at',  ],
#axis=1)

articles = articles['description']
#articles = " ".join(articles.astype(str))

# Remove punctuation
articles = \
articles.map(lambda x: re.sub('[,.!?]', '', x))

# Convert the titles to lowercase
articles = \
articles.map(lambda x: x.lower())

stop_words = stopwords.words('english')
stop_words.extend(['test', 'tests', 'testing', 'every', 'minutes', 'time', 'one', 'two', 'three', 'see', 'met', 'part', 'possible', 'still', 'way', 'says', 'keep', 'tldr', 'first', 'using', 'actually', 'answer', 'bangalore', 'often', 'move', 'real', 'across', 'small', 'new', 'made', 'team', 'may', 'like', 'whether', 'someone', 'question', 'buy', 'looks', 'look' ])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


#wordcloud eda

text_data = ','.join(articles.astype(str))
st_words = set(STOPWORDS)
more_stopwords = {'test', 'tests', 'testing', 'every', 'minutes', 'time', 'one', 'two', 'three', 'see', 'met', 'part', 'possible', 'still', 'way', 'says', 'keep', 'tldr', 'first', 'using' }
st_words = st_words.union(more_stopwords)
text_data = ' '.join(word for word in text_data.split() if word not in st_words)
wcl = WordCloud().generate(text_data)
st.dataframe(df)

plt.imshow(wcl, interpolation='bilinear')
plt.axis("off")
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(wcl)
plt.axis("off")
st.pyplot(fig)

#lda model blogs
data = articles.tolist()
data_words = list(sent_to_words(data))

data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

id2word = corpora.Dictionary(data_words)
texts = data_words
corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[:1][0][:30])



if __name__ == '__main__':
    num_topics = 12
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # if 1 == 1:
    #     LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    #     html_string = pyLDAvis.prepared_data_to_html(LDAvis_prepared)
    #     components.v1.html(html_string, height=800, scrolling=True)


#



# vec = CountVectorizer(stop_words='english')
# dtm = vec.fit_transform(df['description'])
# lda = LatentDirichletAllocation(n_components=5, max_iter=5, random_state=0)
# lda.fit(dtm)
#
# words = vec.get_feature_names_out()
# for topic_idx, topic in enumerate(lda.components_):
#     print(f"\nTopic #{topic_idx + 1}:")
#     top_word_indices = topic.argsort()[:-6:-1]
#     top_words = [words[i] for i in top_word_indices]
#     print(" ".join(top_words))

