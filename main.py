from __future__ import absolute_import

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
from nltk import FreqDist
from nltk.translate import *
from nltk.translate.meteor_score import meteor_score as meteor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from pprint import pprint
#

nltk.download('punkt')
nltk.download('punkt_tab')

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
#articles = articles.drop(columns=['type_of', 'id', 'readable_publish_date', 'slug', 'path', 'url', 'comments_count', 'public_reactions_count', 'collection_id', 'published_timestamp', 'language', 'subforem_id', 'positive_reactions_count', 'cover_image', 'social_image', 'canonical_url', 'created_at', 'edited_at', 'crossposted_at', 'published_at', 'last_comment_at', 'reading_time_minutes', 'tag_list', 'user', 'organization', 'flare_tag'])

articles = articles['description']
#articles = " ".join(articles.astype(str))

# Remove punctuation
articles = \
articles.map(lambda x: re.sub('[,.!?]', '', x))

# Convert the titles to lowercase
articles = \
articles.map(lambda x: x.lower())

stop_plus = ['write', 'work', 'future', 'career', 'evidence', 'essential' ,'page','last' ,'fail' ,'week' ,'guide', 'quality','agents' ,'⭐️⭐️' ,'build', 'built', 'building', 'coding' ,'code', '2026', 'software', 'test', 'tests', 'testing', 'every', 'minutes', 'time', 'one', 'two', 'three', 'see', 'met', 'part', 'possible', 'still', 'way', 'says', 'keep', 'tldr', 'first', 'using', 'actually', 'answer', 'bangalore', 'often', 'move', 'real', 'across', 'small', 'new', 'made', 'team', 'may', 'like', 'whether', 'someone', 'question', 'buy', 'looks', 'look' ]
stop_words = stopwords.words('english')
stop_words.extend(stop_plus)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


#wordcloud eda----------------------------------------------------------------------------------------------------------

text_data = ','.join(articles.astype(str))
st_words = set(STOPWORDS)
#more_stopwords = {'software', 'test', 'tests', 'testing', 'every', 'minutes', 'time', 'one', 'two', 'three', 'see', 'met', 'part', 'possible', 'still', 'way', 'says', 'keep', 'tldr', 'first', 'using' }
more_stopwords = stop_plus
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

#end------------------------------------------------------------------------------------------------------------------

#lda model blogs
data = articles.tolist()
data_words = list(sent_to_words(data))

data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

id2word = corpora.Dictionary(data_words)
texts = data_words
corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[:1][0][:30])



#if __name__ == '__main__':
#    num_topics = 12
#    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
#    pprint(lda_model.print_topics())
#    doc_lda = lda_model[corpus]

    # if 1 == 1:
    #     LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    #     html_string = pyLDAvis.prepared_data_to_html(LDAvis_prepared)
    #     components.v1.html(html_string, height=800, scrolling=True)
#lda end---------------------------------------------------------------------------------------------------------------

#top 10 topics
filtered_words = []

blog_data = ','.join(articles.astype(str))
tokenised = word_tokenize(blog_data)

for token in tokenised:
    if token not in stop_words:
        if len(token) > 3:
            filtered_words.append(token)
fdist = FreqDist(filtered_words)
fdist_top15 = fdist.most_common(15)
st.dataframe(fdist_top15)
#-----------------------------------------------------------------------------------------------------------------------

#tf-idf


#tfidf_vector = TfidfVectorizer(stop_words='english')
#tfidf_result = tfidf_vector.fit_transform(tokenised)









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

