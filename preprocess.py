import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist

import numpy as np
import pandas as pd
import os
import shutil

DATA_ROOT = "data/"

def get_corpus_and_basic_stats(root):
    # This regex excludes files of the form "*.meta.txt" while selecting files of the form "*.txt"
    corpus = PlainTextCorpusReader(root, r"[0-9]+(?!meta)\.txt")
    print("Creating frequency distribution")
    freqdist = FreqDist(corpus.words())

def create_train_test_split(test_ratio=0.3, random_seed=9001):
    summary_df = pd.read_csv(DATA_ROOT+"summary.csv", index_col=0)
    
    #Ensure the data/train and data/test directories are created
    train_dir = DATA_ROOT + "train/"
    test_dir = DATA_ROOT + "test/"
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    rng = np.random.default_rng(seed=random_seed)
    N = len(summary_df.index)
    split = int((1-test_ratio)*N) + 1
    perm = rng.permutation(summary_df.index.values)
    train_sample = perm[:split]
    test_sample = perm[split:]
    
    for text_id in train_sample:
        filename = f"{text_id}.txt"
        meta_filename = f"{text_id}.meta.txt"
        
        shutil.copy2(DATA_ROOT + filename, train_dir + filename)
        shutil.copy2(DATA_ROOT + meta_filename, train_dir + meta_filename)
        
    for text_id in test_sample:
        filename = f"{text_id}.txt"
        meta_filename = f"{text_id}.meta.txt"
        
        shutil.copy2(DATA_ROOT + filename, test_dir + filename)
        shutil.copy2(DATA_ROOT + meta_filename, test_dir + meta_filename)