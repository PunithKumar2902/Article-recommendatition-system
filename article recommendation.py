#importing necessary libraries

import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

#visualize the data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/articles.csv", encoding='latin1')
data.head()

#generating recommendations

articles = data["Article"].tolist()
uni_tfidf = text.TfidfVectorizer(input=articles, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(articles)
uni_sim = cosine_similarity(uni_matrix)
def recommend_articles(x):
    return ", ".join(data["Title"].loc[x.argsort()[-5:-1]])    
data["Recommended Articles"] = [recommend_articles(x) for x in uni_sim]
data.head()
