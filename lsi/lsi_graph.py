from lsi import LSI
import numpy as np
import networkx as nx
import pandas as pd
from joblib import Parallel, delayed, load

df = pd.read_json('../../data/merged_summary_topics.json')

def build_graph(model):
    n = len(model.Ahat)
    similarity = Parallel(n_jobs=-1, verbose=0)(
        delayed(model.similar)(i) for i in range(n)
    )

    most_similar = [pair[0] for pair in similarity]
    least_similar = [pair[1] for pair in similarity]

    G = nx.Graph()
    for i, j in enumerate(most_similar):
        G.add_edge(i, j)

    return G

def connected_components(model):
    G = build_graph(model)
    cc = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    return cc

def list_component_topics(comp):
    files = ['data/{}.txt'.format(c) for c in list(comp)]
    topic_lists = []
    for file in files:
        topics = df[df.File == file].topic_lists.values
        if len(topics) > 0:
            topic_lists.append(topics[0])

    return topic_lists

def list_component_speakers(comp):
    files = ['data/{}.txt'.format(c) for c in list(comp)]
    speaker_list = []
    for file in files:
        speaker = df[df.File == file].Speaker.values
        if len(speaker) > 0:
            speaker_list.append(speaker[0])

    return speaker_list
