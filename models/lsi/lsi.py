# lsi.py

import os
import sys
import string
import numpy as np
from math import log
from scipy import sparse
from scipy import linalg as la
from collections import Counter
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

USAGE = "USAGE: python lsi.py [talk file paths]"

def similar(i, Xhat):
    """
    Takes an index and matrix representing the principal components and returns the indices of
    the documents that are the most and least similar to i using cosine similarity.
    
    Parameters:
    i index of a document
    Xhat decomposed data
    
    Returns:
    index_max: index of the document most similar to document i
    index_min: index of the document least similar to document i
    """
    X_i = Xhat[i]
    X_i_norm = la.norm(X_i)

    # set up benchmarks
    index_max = i
    max_similarity = -1.0
    index_min = i
    min_similarity = 1.0

    for j in range(len(Xhat)):
        # skip the i == j becase it has similarity 1 with itself
        if i != j:
            similarity = np.dot(X_i, Xhat[j]) / (X_i_norm * la.norm(Xhat[j]))

            # update cosine similarity benchmarks as needed
            if similarity > max_similarity:
                index_max = j
                max_similarity = similarity
            
            if similarity < min_similarity:
                index_min = j
                min_similarity = similarity
    
    return index_max, index_min
    
def document_converter():
    '''
    Converts talk documents into an n by m array where m is the number 
    of vocabulary words and n is the number of documents
    
    Returns:
    X sparse matrix (n x m): Each row represents a document
    paths (list): list where each element is a talk path eg: path[0] is './Addresses/1990-Bush.txt'
    '''
    # Get list of filepaths to each text file in the folder.
    folder = "./data/"
    paths = [folder+p for p in os.listdir(folder) if p.endswith(".txt")]

    # Helper function to get list of words in a string.
    def extractWords(text):
        ignore = string.punctuation + string.digits
        cleaned = "".join([t for t in text.strip() if t not in ignore])
        return cleaned.lower().split()

    # Initialize vocab set, then read each file and add to the vocab set.
    vocab = set()
    for p in paths:
        with open(p, 'r', encoding="utf8") as infile:
            for line in infile:
                vocab.update(extractWords(line)) #union sets together


    # load stopwords
    with open("stopwords.txt", 'r',  encoding="utf8") as f:
        stops = set([w.strip().lower() for w in f.readlines()])

    # remove stopwords from vocabulary, create ordering
    vocab = {w:i for i, w in enumerate(vocab.difference(stops))}


    counts = []      # holds the entries of X
    doc_index = []   # holds the row index of X
    word_index = []  # holds the column index of X

    # Iterate through the documents.
    for doc, p in enumerate(paths):
        with open(p, 'r', encoding="utf8") as f:
            # Create the word counter.
            ctr = Counter()
            for line in f:
                ctr.update(extractWords(line))
            # Iterate through the word counter, store counts.
            for word, count in ctr.items():
                if word in vocab:
                    word_index.append(vocab[word])
                    counts.append(count)
                    doc_index.append(doc)

    # Create sparse matrix holding these word counts.
    X = sparse.csr_matrix((counts, [doc_index, word_index]),
                            shape=(len(paths), len(vocab)), dtype=np.float64)
    return X, paths

def weighted_document_converter():
    '''
    Converts talk documents into an n by m array where m is the number 
    of vocabulary words and n is the number of documents. It gives weights
    to the most important words in the vocabulary.
    
    Returns:
    A (sparse matrix, n x m): Each row represents a document
    paths (list): list where each element is a talk path eg: path[0] is './Addresses/1990-Bush.txt'
    '''
    # Get list of filepaths to each text file in the folder.
    folder = "./data/"
    paths = [folder+p for p in os.listdir(folder) if p.endswith(".txt")]

    # Helper function to get list of words in a string.
    def extractWords(text):
        ignore = string.punctuation + string.digits
        cleaned = "".join([t for t in text.strip() if t not in ignore])
        return cleaned.lower().split()

    # Initialize vocab set, then read each file and add to the vocab set.
    vocab = set()
    for p in paths:
        with open(p, 'r', encoding="utf8") as infile:
            for line in infile:
                vocab.update(extractWords(line)) #union sets together


    # load stopwords
    with open("stopwords.txt", 'r',  encoding="utf8") as f:
        stops = set([w.strip().lower() for w in f.readlines()])

    # remove stopwords from vocabulary, create ordering
    vocab = {w:i for i, w in enumerate(vocab.difference(stops))}

    t = np.zeros(len(vocab))
    counts = []      # holds the entries of X
    doc_index = []   # holds the row index of X
    word_index = []  # holds the column index of X

    # iterate through the documents.
    for doc, p in enumerate(paths):
        with open(p, 'r', encoding="utf8") as f:
            # Create the word counter.
            ctr = Counter()
            for line in f:
                ctr.update(extractWords(line))
            # Iterate through the word counter, store counts.
            for word, count in ctr.items():
                if word in vocab:
                    word_ind = vocab[word]
                    word_index.append(word_ind)
                    counts.append(count)
                    doc_index.append(doc)
                    t[word_ind] += count
                    
    # get global weights
    t = np.array(t)
    X = sparse.csr_matrix((counts, [doc_index, word_index]),
                           shape=(len(paths), len(vocab)), dtype=np.float64).toarray()

    # calculate p as a matrix and g as a row vector
    P = X / t
    G = P * np.log(P + 1)
    g = 1 + (np.sum(G, axis=0) / len(paths))

    # calculate A as a sparse matrix
    A = sparse.csr_matrix(g * np.log(X + 1))
    
    return A, paths


def cross_similarity(talks, n_components=7):
    """
    Uses LSI, applied to the globally weighted word count matrix A, with the
    first 7 principal components to find the most similar and least similar talkes

    Parameters:
        talk str: Path to talk eg: "./data/1002.txt"
        (int): Number of principal components

    Returns:
        tuple of str: (Most similar talk, least similar talk)
    """
    A, paths = weighted_document_converter()

    pca = PCA(n_components=n_components)
    Ahat = pca.fit_transform(A.toarray())

    def single_talk_similarity(talk_path):
        # determine the index of the talk of choice
        try:
            i = paths.index(talk_path)
        except ValueError:
            raise ValueError(f"No such talk '{talk_path}' found")

        # find index and corresponding path of most/least similar talkes
        index_max, index_min = similar(i, Ahat)
        talk_max = paths[index_max]
        talk_min = paths[index_min]

        return talk_max, talk_min

    cross_similarities = Parallel(n_jobs=-1, verbose=0)(
        delayed(single_talk_similarity)(talk) for talk in talks
    )

    return cross_similarities

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print(USAGE)
        exit

    talk_paths = sys.argv[1:]
    cross_similarities = cross_similarity(talk_paths)

    for i, talk_path in enumerate(talk_paths):
        print(talk_path, '::', cross_similarities[i])
