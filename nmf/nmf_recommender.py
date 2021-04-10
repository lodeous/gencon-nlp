import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

import ast

DATA_ROOT = "../data/"

class NMFactorizer:
    
    def __init__(self):
        pass
        
    def train(self, V, start_rank=3, max_iter=1000, rtol=1e-4, max_rank=np.inf, verbose=False):
        """Train a Nonnegative matrix factorization on the nonnegative matrix V.
        This progressively increases the rank of the factorization if it does not meet the relative
        tolerance standard. The results are stored as members of the class"""
        
        #This is very similar to the lab code we used for problem 5
        rank = start_rank
        nmf = NMF(init='random', max_iter=max_iter)
        benchmark = np.linalg.norm(V)*rtol
        
        if verbose:
            print(f"Attempting to converge to residual less than {benchmark:.6f}")
        
        done = False
        success = False
        while not done:
            nmf.set_params(n_components=rank)
            W = nmf.fit_transform(V)
            H = nmf.components_
            Vhat = W@H
            rmse = np.sqrt(mean_squared_error(V, Vhat))
            if verbose:
                print(f"Rank={rank} factorization rmse={rmse}")
                
            if rmse < benchmark:
                done = True
                success = True
            else:
                rank += 1
                
            if rank > max_rank:
                done = True
                success = False
                
        print(f"Found a tolerable NMF factorization with rank={rank}")
        #save results to the object
        self.success = success
        self.nmf = nmf
        self.W = W
        self.H = H
        self.Vhat = Vhat
        self.rank = rank
        #Mask out positive values of V which have "already been seen"
        #Allowing us to make good recommendations
        self.masked_Vhat = Vhat * (V == 0)
        
        return success

class NMFRecommender:

    def __init__(self):
        self.talk_nmf = NMFactorizer()
        self.speaker_nmf = NMFactorizer()
        pass
        
    def load_data(self, min_talks=2, csv_file="caleb_merged_topics.csv"):
        """Load the topic and speaker data for the top topics and speakers"""
        talk_data = pd.read_csv(DATA_ROOT + csv_file, index_col=0)
        #Drop talks which don't have topics
        talk_data.dropna(subset=["Topics"], inplace=True)
        #Map the stringified topic lists to python lists; keep dtype as object
        talk_data["Topics"] = talk_data["Topics"].map(ast.literal_eval)
        
        #Find unique topics and topic counts (in case we want to filter to only popular topics
        self.topic_count = {}
        for topics in talk_data["Topics"]:
            for topic in topics:
                if topic in self.topic_count:
                    self.topic_count[topic] += 1
                else:
                    self.topic_count[topic] = 1
        
        self.topic_indices = {topic:i for i, topic in enumerate(self.topic_count)}
        self.topic_lookup = {i:topic for i, topic in enumerate(self.topic_count)}
        talk_ids = []
        
        #Create a nonnegative matrix with talks as rows representing talk-choices as features
        #and topics as columns representing potential users interested in those topics
        nrows = len(talk_data.index)
        ncols = len(self.topic_indices)
        self.V_talks = np.zeros((nrows, ncols))
        self.talk_ids = np.empty(nrows, dtype=int)
        
        for row, (talk_id, topics) in enumerate(zip(talk_data.index, talk_data["Topics"])):
            self.talk_ids[row] = talk_id
        
            for topic in topics:
                topic_col = self.topic_indices[topic]
                self.V_talks[row, topic_col] = 1
        
        # I want to include my favorite Japanese General authority
        #Elder Kazuhiko Yamashita who has only given 2 talks. I don't think adding
        #speakers who have spoken few times wil affect the factorization/recommendations to other
        #speakers very much and could provide some useful information. However predictions on speakers
        #with less talks will be less reliable.
        speakers = talk_data["Speaker"].value_counts()
        top_speakers = speakers[speakers >= min_talks]
        
        #Create a nonnegative matrix with speakers as columns and topics as rows
        
        nrows = len(self.topic_count)
        ncols = len(top_speakers.index)

        self.V_speakers = np.zeros((nrows, ncols))

        self.speaker_indices = {speaker:i for i, speaker in enumerate(top_speakers.index)}
        self.speaker_lookup = {i:speaker for i, speaker in enumerate(top_speakers.index)}
        self.speaker_topics = {}

        for i, speaker in enumerate(top_speakers.index):
            sp_topic_counts = {topic:0 for topic in self.topic_count}
            #Count the number of times the speaker has talked on a given topic
            for topics in talk_data.loc[talk_data["Speaker"] == speaker, "Topics"]:
                for topic in topics:
                    sp_topic_counts[topic] += 1
            
            #Store the aggregate topic counts for the speaker in the nonnegative matrix
            for topic in sp_topic_counts:
                topic_row = self.topic_indices[topic]
                self.V_speakers[topic_row, i] = sp_topic_counts[topic]
            
            #We might be interested in what topics they've spoken on in the past
            self.speaker_topics[speaker] = sp_topic_counts
        
        #return talk_data for later lookup
        return talk_data
    
    def train_topics_vs_talks(self, **kwargs):
        return self.talk_nmf.train(self.V_talks, **kwargs)
    
    def train_speakers_vs_topics(self, **kwargs):
        return self.speaker_nmf.train(self.V_speakers, **kwargs)
        
    def recommend_talks(self, topic, n_talks=10):
        """
        Recommend the top n_talks talks from the Nonnegative Matrix Factorization.
        
        Returns:
            top_talk_ids (list) a list of tuples, n_talks long, containing the ID of the recommended talk
        """
        topic_index = self.topic_indices[topic]
        top_indices = self.talk_nmf.masked_Vhat[:, topic_index].argsort()[::-1][:n_talks]
        return [self.talk_ids[index] for index in top_indices]
        
    def recommend_topics_to_speaker(self, speaker, n_topics=10):
        speaker_index = self.speaker_indices[speaker]
        top_indices = self.speaker_nmf.masked_Vhat[:, speaker_index].argsort()[::-1][:n_topics]
        return [self.topic_lookup[index] for index in top_indices]
    
    #def recommend_speakers_from_topic()
    