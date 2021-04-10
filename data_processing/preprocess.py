import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist

import numpy as np
import pandas as pd
import os
import shutil

DATA_ROOT = "../data/"

def get_corpus_and_basic_stats(root):
    # This regex excludes files of the form "*.meta.txt" while selecting files of the form "*.txt"
    corpus = PlainTextCorpusReader(root, r"[0-9]+(?!meta)\.txt")
    print("Creating frequency distribution")
    freqdist = FreqDist(corpus.words())
    
def discover_train_test_split_files():
    train_list = []
    
    with os.scandir(DATA_ROOT + "train") as it:
        for entry in it:
            if entry.is_file():
                if entry.name.endswith('.meta.txt'):
                    pass
                elif entry.name.endswith('.txt'):
                    train_list.append(entry.name)
    
    test_list = []
    with os.scandir(DATA_ROOT + "test") as it:
        for entry in it:
            if entry.is_file():
                if entry.name.endswith('.meta.txt'):
                    pass
                elif entry.name.endswith('.txt'):
                    test_list.append(entry.name)
                    
    with open(DATA_ROOT + "train_split.info", 'w') as f:
        for filename in train_list:
            f.write(filename + '\n')
    with open(DATA_ROOT + "test_split.info", 'w') as f:
        for filename in test_list:
            f.write(filename + '\n')
    
    return train_list, test_list
            
def get_train_test_split():
    """Returns a list of files in the training split and one of files in the test split."""
    if not os.path.exists(DATA_ROOT + "train_split.info"):
        return discover_train_test_split_files()
    
    with open(DATA_ROOT + "train_split.info") as f:
        train_list = [line.strip() for line in f]
    
    with open(DATA_ROOT + "test_split.info") as f:
        test_list = [line.strip() for line in f]
        
    return train_list, test_list
    
def mark_train_test_in_dataframe(df):
    """For a Pandas DataFrame with the column "File", this method modifies in place the given
    DataFrame to have a column "Train" with a value of 1 signalling if the file is in the
    training set and 0 if its in the test set."""
    train_list, test_list = get_train_test_split()
    
    #Create new column marking the split
    df["Train"] = 1
    
    for filename in test_list:
        df.loc[df["File"] == "data/"+filename, "Train"] = 0
    
def create_train_test_split(test_ratio=0.3, random_seed=9001):
    summary_df = pd.read_csv(DATA_ROOT+"../summary.csv", index_col=0)
    
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