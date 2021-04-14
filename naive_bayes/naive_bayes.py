# naive_bayes.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def initialize(column, data_relative='../', train_size=0.7, random_state=42):
    df = pd.read_json('../data/merged_summary_topics.json')
    paths = (data_relative + df.File).values
    y = df[column]

    cv = CountVectorizer(input='filename', stop_words='english')
    X = cv.fit_transform(paths)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    return (X_train, X_test, y_train, y_test), paths, cv
