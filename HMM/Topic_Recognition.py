import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm
import string
from gensim import corpora
from gensim.utils import simple_preprocess
from utils import prep_text, vec_translate, prep_data
import pickle

def go_through_topics():
    data = pd.read_csv("../merged_summary_topics.csv")
    topic_list = [column for column in data.columns if column not in
                  ['Year', 'Speaker', 'Title', 'File', 'Month', 'topic_lists', 'Train',
                   'Month_letter', 'Month', 'Kicker', 'Unnamed: 0', 'Unnamed: 0.1']]
    topics_used = []
    test_data = None
    for topic in topic_list:
        file_name = ('_').join(topic.split(' '))
        df = pd.read_csv(f"../Topic_Data/{file_name}.csv")
        #use only topics that have more than 50 talks
        if df.shape[0] >=50:
            topics_used.append(topic)
        else:
            continue

        df_train = df[df['Train'] == 1]
        df_test = df[df['Train'] == 0]
        if test_data is None:
            test_data = df_test
        else:
            test_data.append(df_test)
        #read in all the talks on topic topic
        df_talks = []
        #go through all the talks in the training set
        for filename in df_train["File"]:
            with open("../" + filename, "r") as f:
                text = f.read()
                processed = simple_preprocess(text)
                if len(text):
                    df_talks.append(processed)
        #concatenate the talks
        df_text = sum(df_talks, start=[])

        #create the dictionary
        dictionary = corpora.Dictionary([df_text])

        #minimize the aic to choose the optimal number of components
        components, AIC = hyperparameter_states(df_text, dictionary, np.arange(2, 6), df_talks)

        #create the best model
        best_model = hmm.MultinomialHMM(n_components=components, n_iter=100)
        #train the model
        best_model.fit(prep_text(df_text, dictionary))

        #save the model
        with open(f"{topic}bestModel", 'wb') as file:
            pickle.dump(best_model, file)
    return topic_list, test_data

def hyperparameter_states(text, dictionary, list_of_states, talks, n=50):
    def calculate_aic(n, mse, num_params):
        aic = n * mse + 2 * num_params
        return aic
    best_aic = np.inf
    best_state = None
    for num in list_of_states:
        model = hmm.MultinomialHMM(n_components=num, n_iter=100, tol=1e-3)
        model.fit(prep_text(text, dictionary))
        score = model.score(prep_text(talks[-1], dictionary))
        if score < best_aic:
            best_aic = score
            best_state = num
    return best_state, best_aic

topic_list, test_data = go_through_topics()

with open('topic_list.txt', 'a') as file:
    file.write(str(topic_list))
test_data.to_json('topic_hmm_test_data.json')
