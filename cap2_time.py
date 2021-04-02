import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import copy
from bs4 import BeautifulSoup
import unicodedata
from collections import Counter
import re
from pprint import pprint
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB

pokemon_df_complete = pd.read_csv('~/DSI_ii/cap_stones/DSI_CS2/data/pokemongo_new.csv')
pokemon_df = pokemon_df_complete[pokemon_df_complete.created<'2021-03-29 10:49:12']
pokemon_df.drop(pokemon_df.columns[0],axis=1)
pokemon_df['low_score'] = pokemon_df['score'] <=1

title_df = pokemon_df['title']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(title_df)
y = pokemon_df['low_score']

n_b_multi = SKMultinomialNB()
n_b_multi.fit(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y)



title_nb_m_train_s = []
title_nb_m_test_s = []

for _ in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    n_b_multi.fit(X,y)
    title_nb_m_train_s.append(n_b_multi.score(X_train, y_train))
    title_nb_m_test_s.append(n_b_multi.score(X_test, y_test))

print(sum(title_nb_m_train_s)/(1000))
print(sum(title_nb_m_test_s)/1000)