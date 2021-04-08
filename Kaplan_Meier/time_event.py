import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

summary = pd.read_csv('summary.csv', header=0, index_col=0)

def get_list_of_words(filename):
    with open(filename, 'r') as file:
        text = file.read()
        lines = text.split('\n')
        words_lists = [line.split(" ") for line in lines]
        words = [word for word_list in words_lists for word in word_list if len(word_list) > 1]
        return words

def get_time_event(filename, event):
    words = get_list_of_words(filename)
    event_list = np.array([0 if word != event else 1 for word in words])
    if np.sum(event_list) == 0:
        outcome = 0
        time = len(event_list)
    else:
        outcome = 1
        time = np.argmax(event_list)
    return time, outcome

def get_time_event_array(event):
    summary = pd.read_csv('summary.csv', header=0, index_col=0)
    list_of_files = summary['File'].to_numpy()
    time_event = [get_time_event(file_, event) for file_ in list_of_files]
    time = [time[0] for time in time_event]
    outcome = [evnt[1] for evnt in time_event]
    summary['Time'] = time
    summary['Event'] = outcome
    summary.to_csv('event_time.csv')

if __name__ == "__main__":
    get_time_event_array('Christ')
