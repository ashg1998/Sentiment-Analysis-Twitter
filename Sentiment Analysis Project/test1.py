# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 01:30:52 2020

@author: user
"""

import pandas as pd
data  = pd.read_csv('train.csv')
data.head()
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

corpus =[]
ps =PorterStemmer()
for i in range(0,len(data)):
    reviews = re.sub('[^a-zA-Z]' , ' ', data['tweet'][i])
    reviews = reviews.lower()
    reviews = reviews.split()
    reviews = [ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviews = ' '.join(reviews)
    corpus.append(reviews)
clean_data = pd.DataFrame(corpus , columns=['tweet'])
clean_data['label'] = data['label']
clean_data['id'] = data['id']

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()

clean_data['tweet'] = cv.fit_transform(clean_data['tweet'])

X = clean_data['tweet']
x= cv.fit_transform(corpus)
