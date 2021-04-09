import numpy as np
import string


# convert a talk into an array of integers using the dictionary provided
def prep_text(text, dictionary):
    X = dictionary.doc2idx(text)
    return np.array(X).reshape(-1, 1)

#
# Character-level utility functions from homework 9.5
#

def vec_translate(a, my_dict):
    # translate array from symbols to state numbers or -vice versa
    return np.vectorize(my_dict.__getitem__)(a)


def prep_data(filename):
    # Get the data as a single string
    with open(filename) as f:
        data=f.read().lower() #read and convert to -lower case
    # remove punctuation and newlines
    remove_punct = {ord(char): None for char in string.punctuation+"\n\r"}
    data = data.translate(remove_punct)
    # make a list of the symbols in the data
    symbols = sorted(list(set(data)))
    # convert the data to a NumPy array of symbols
    a = np.array(list(data))
    #make a conversion dict from symbols to state -numbers
    symbols_to_obs = {x:i for i,x in enumerate(symbols)}
    #convert the symbols in a to state numbers
    obs_sequence = vec_translate(a,symbols_to_obs)
    return symbols, obs_sequence
