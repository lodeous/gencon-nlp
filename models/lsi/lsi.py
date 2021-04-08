# lsi.py

import string
import numpy as np
from scipy import sparse
from scipy import linalg as la
from collections import Counter
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

class LSI():
    """
    A PCA-based LSI model for finding similarities between documents.
    
    Attributes:
        n_components (int): The number of principle components to use (default: 7)
        stopwords_file (str): The path the a file of stopwords to use (default: './stopwords.txt)
        encoding (str): The file encoding of the documents (default: 'utf8')
    """

    def __init__(self, n_components=7, stopwords_file='./stopwords.txt', encoding='utf8'):
        self.n_components = n_components
        self.stopwords_file = stopwords_file
        self.encoding = encoding


    def fit(self, paths):
        """
        Fits the model with documents found in the given paths.

        Parameters:
            paths (list(str)): list of paths of the documents
        """
        self._A, self.paths = self._weighted_document_converter(paths)

        self.pca = PCA(n_components=self.n_components)
        self.Ahat = self.pca.fit_transform(self._A.toarray())


    def similar(self, i):
        """
        Determines the most similar and least similar document to the given
        index using cosing similarity.

        Parameters:
            i (int): index of document to compare to

        Returns:
            tuple(str): (Most similar document, least similar document)
        """
        index_max, index_min = self._cosine_similarity(i, self.Ahat)
        return index_max, index_min


    def cross_similarity(self, doc_paths):
        """
        Uses LSI with the first n_components principal components to find the
        most similar and least similar talkes

        Parameters:
            doc_paths: Path to documents eg: "./data/1002.txt"

        Returns:
            tuple(str): (Most similar document, least similar document)
        """
        def single_document_similarity(doc_path):
            # determine the index of the document of choice
            try:
                i = self.paths.index(doc_path)
            except ValueError:
                raise ValueError(f"No such document '{doc_path}' found")

            # find index and corresponding path of most/least similar documents
            index_max, index_min = self.similar(i)
            document_max = self.paths[index_max]
            document_min = self.paths[index_min]

            return document_max, document_min

        cross_similarities = Parallel(n_jobs=-1, verbose=0)(
            delayed(single_document_similarity)(doc_path) for doc_path in doc_paths
        )

        return cross_similarities


    def _cosine_similarity(self, i, Xhat):
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


    def _weighted_document_converter(self, paths):
        '''
        Converts talk documents into an n by m array where m is the number 
        of vocabulary words and n is the number of documents. It gives weights
        to the most important words in the vocabulary.
        
        Returns:
        A (sparse matrix, n x m): Each row represents a document
        paths (list): list where each element is a doc path eg: path[0] is './data/1001.txt'
        '''
        # Helper function to get list of words in a string.
        def extractWords(text):
            ignore = string.punctuation + string.digits
            cleaned = "".join([t for t in text.strip() if t not in ignore])
            return cleaned.lower().split()

        # Initialize vocab set, then read each file and add to the vocab set.
        vocab = set()
        for p in paths:
            with open(p, 'r', encoding=self.encoding) as infile:
                for line in infile:
                    vocab.update(extractWords(line)) #union sets together

        # load stopwords
        with open(self.stopwords_file, 'r',  encoding=self.encoding) as f:
            stops = set([w.strip().lower() for w in f.readlines()])

        # remove stopwords from vocabulary, create ordering
        vocab = {w:i for i, w in enumerate(vocab.difference(stops))}

        t = np.zeros(len(vocab))
        counts = []      # holds the entries of X
        doc_index = []   # holds the row index of X
        word_index = []  # holds the column index of X

        # iterate through the documents.
        for doc, p in enumerate(paths):
            with open(p, 'r', encoding=self.encoding) as f:
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
                            shape=(len(paths), len(vocab)), dtype=np.float64)#.toarray()

        # calculate p as a matrix and g as a row vector
        P = X / t
        logP = sparse.csr_matrix(np.log(P + 1))
        P = sparse.csr_matrix(P)
        G = P.multiply(logP)
        del P; del logP

        g = 1 + np.array((np.sum(G, axis=0) / len(paths))).flatten()
        del G

        # calculate A as a sparse matrix
        A = sparse.csr_matrix(g * np.log(X.toarray() + 1))
        
        return A, paths
