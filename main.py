from __future__ import absolute_import

import base64
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
import streamlit.components.v1 as v1components
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
st.set_page_config(layout="wide")


st.title(":color[Software Testing Topic Digest]{foreground=#20b2aa}", text_alignment="center")
st.header(":color[What is being discussed in the industry?]{foreground=#008b8b}", text_alignment="left")
devlink = 'https://dev.to/t/testing'
st.markdown('''1000 blog posts about software testing from the blogging platform [DEV.to](%s) were analysed to narrow down the most discussed topics in the software testing industry.''' %devlink)
col1, col2 = st.columns([1, 1], gap='small', border=False)

#api
URL = 'https://dev.to/api/articles'
params = {
    'tag' : 'testing',
    'page' : '1',
    'per_page' : '1000'
}

response = requests.get(url=URL, params=params)
response.raise_for_status()
articles = response.json()

df = pd.DataFrame(articles)


df.to_csv('articles.csv', index=False)
articles = pd.read_csv('articles.csv')

articles = articles['description']
#articles = " ".join(articles.astype(str))

articles = \
articles.map(lambda x: re.sub('[,.!?]', '', x))

articles = \
articles.map(lambda x: x.lower())

lemmatizer = WordNetLemmatizer()


stop_plus = ['start' ,'constantly','free', 'without', 'write', 'work', 'future', 'career', 'evidence', 'essential' ,'page','last' ,'week', 'weeks' ,'guide', 'quality','⭐️⭐️' ,'build', 'built', 'building', 'coding' ,'code', '2026', 'software', 'test', 'tests', 'testing', 'every', 'minutes', 'time', 'one', 'two', 'three', 'see', 'met', 'part', 'possible', 'still', 'way', 'says', 'keep', 'tldr', 'first', 'using', 'actually', 'answer', 'bangalore', 'often', 'move', 'real', 'across', 'small', 'new', 'made', 'team', 'may', 'like', 'whether', 'someone', 'question', 'buy', 'looks', 'look', 'need', 'something', 'know', 'telegram', 'use', 'using', 'never', 'nothing', 'right', 'thing', 'dr', 'tl', 'open', 'five', 'strength', 'hundreds', 'get', 'best', 'post', 'elevate', 'checks', 'change', 'problem' ]
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

articles_lem = []

blog_data = ','.join(articles.astype(str))
tokenised = word_tokenize(blog_data)
tagged_tokens = pos_tag(tokenised)

for word, tag in tagged_tokens:
        articles_lem.append(
            lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
tokenised = word_tokenize(' '.join(articles_lem))

#wordcloud eda----------------------------------------------------------------------------------------------------------
text_data = remove_stopwords(tokenised)
#text_data = ','.join(text_data)
# st_words = set(STOPWORDS)
# more_stopwords = stop_plus
# st_words = st_words.union(more_stopwords)
# text_data = ' '.join(word for word in text_data.split() if word not in st_words)
wcl = WordCloud().generate(text_data)
#st.dataframe(df)

plt.imshow(wcl, interpolation='bilinear')
plt.axis("off")
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(wcl)
plt.axis("off")
with col1:
    with st.container(width=800):
        st.markdown(''':color[Topic wordcloud]{foreground=#48d1cc}''')
        st.pyplot(fig)

#end------------------------------------------------------------------------------------------------------------------

#lda model blogs------------------------------------------------------------------------------------------------------

html_lda = open('blogs_lda.html', 'r')
raw_html = html_lda.read().encode("utf-8")
raw_html = base64.b64encode(raw_html).decode()
st.markdown(''':color[LDA (Latent Dirichlet Allocation) model visualisation, illustrating term distribution within identified topics. Saliency refers to importance of terms across the whole text corpus, while relevance refers to importance of terms within each topic]{foreground=#48d1cc}''')
v1components.iframe(f"data:text/html;base64,{raw_html}", height=800, width=1260)
#components.v1.html('blogs_lda.html', height=800, scrolling=True)

#lda end---------------------------------------------------------------------------------------------------------------

#top 15 topics---------------------------------------------------------------------------------------------------------
filtered_words = []

for token in tokenised:
    if token not in stop_words:
        if len(token) > 3:
            filtered_words.append(token)
fdist = FreqDist(filtered_words)
fdist_top15 = fdist.most_common(15)
fdist_top15_df = pd.DataFrame(fdist_top15, columns=['Topic', 'Times mentioned'])
with col2:
    with st.container(width=700):
        st.markdown(''':color[Top 15 topics discussed in software testing blogs]{foreground=#48d1cc}''')
        st.dataframe(fdist_top15_df, hide_index=True)
#-----------------------------------------------------------------------------------------------------------------------

#tf-idf

# topiclist = vectorizer.get_feature_names_out()
# print(topiclist)
# tfidf_blogs.toarray()
#
# s = ''
#
# for i in topiclist:
#     s += "- " + i + "\n"
#
# st.markdown(s)

syllabi_df = pd.DataFrame(columns=['syllabus', 'keywords'])
#result_df = pd.DataFrame(columns=['Topic', 'Found in:'])
syl_corpus = []

#tf-idf on syllabi--------------------------------------------------------------------------------------------
input_folder = "./ISTQB_json"

for file in os.listdir(input_folder):
    if file.lower().endswith(".json"):
        json_path = os.path.join(input_folder, file)

        with open(json_path, 'r', encoding='utf-8') as json_file:
            syllabus = json.load(json_file)

        syl_corpus.append(syllabus)

        syllabus_text = re.sub(
            r'[^A-Za-z\s]',
            ' ',
            str(syllabus)
        )

        tokenised_syl = word_tokenize(syllabus_text)

        syl_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, stop_words=stop_words)
        tfidf_syllabi = syl_vectorizer.fit_transform(tokenised_syl)
        syl_topics = syl_vectorizer.get_feature_names_out()
        filename = os.path.splitext(os.path.basename(json_path))[0]
        #print(filename)
        #print(syl_topics)
        syllabi_df.loc[len(syllabi_df)] = [filename, syl_topics]


#st.dataframe(syllabi_df)

#compare blogs to syllabi

syl_documents = []

for syllabus in syl_corpus:
    syllabus_text = re.sub(r'[^A-Za-z\s]', ' ', str(syllabus))
    syl_documents.append(syllabus_text)

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words=stop_words)
tfidf_all_syllabi = vectorizer.fit_transform(syl_documents)
#vectorizer = TfidfVectorizer(min_df=15, max_df = 0.5, sublinear_tf=True, use_idf =True, stop_words = stop_words)
blog_text = re.sub(r'[^A-Za-z\s]', ' ', str(tokenised))
tfidf_blogs = vectorizer.transform([blog_text])
topic_list = vectorizer.get_feature_names_out()
print(topic_list)

print("Syllabi TF-IDF shape:", tfidf_all_syllabi.shape)
print("Blog TF-IDF shape:", tfidf_blogs.shape)



similarity_matrix = cosine_similarity(tfidf_blogs, tfidf_all_syllabi)[0]

best_indices = similarity_matrix.argsort()[::-1][:24]
best_indices_df = pd.DataFrame(columns=['ISTQB Syllabus', 'Similarity Score'])
for i in best_indices:
    best_indices_df.loc[len(best_indices_df)] = [syllabi_df.iloc[i]["syllabus"], similarity_matrix[i]]
    print(
        syllabi_df.iloc[i]["syllabus"],
        similarity_matrix[i]
    )
print(similarity_matrix)

print("Similarity shape:", similarity_matrix.shape)

st.header(":color[Which teaching materials are aligned with the industry discussions?]{foreground=#008b8b}")
st.text("The International Software Testing Qualifications Board, or ISTQB, is a global software testing certification board that offers the Certified Tester qualification scheme built around syllabi of various depths and areas of testing knowledge. The chart below analyses the similarity of the topics discussed by software testing professionals in blogs to each ISTQB syllabus, and orders the syllabi by similarity score.")
syl_chart = best_indices_df
fig = px.bar(syl_chart,x='Similarity Score', y='ISTQB Syllabus', color='Similarity Score', orientation='h', width=800, height=800,  color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig)

url = "https://istqb.org/certifications"
st.markdown('''Below is a list of the ISTQB syllabi, ordered by relevance according to the similarity calculation illustrated by the chart above. 
      You can find all ISTQB syllabi here:''')
with st.container(width=700):

    st.link_button("ISTQB Certifications", url)
    for index, item in enumerate(best_indices_df["ISTQB Syllabus"], start=1):
        st.markdown(f"{index}. {item}")

with st.container(width=700):
    #df = df.style.background_gradient()
    st.caption('List of blogs used:')
    st.dataframe(df)

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

