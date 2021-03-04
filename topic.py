import requests
import os
from bs4 import BeautifulSoup
import re
from matplotlib import pyplot as plt
import numpy as np
import re
import pandas as pd

def get_title(filename="test.html"):
    """Read the specified file and load it into BeautifulSoup. Return the title tag 
    """
    with open(filename, "r") as my_file:
        file_string = my_file.read()
        file_soup = BeautifulSoup(file_string, 'html.parser')
        #find all of the a tags with href attribute
        title = file_soup.select("title")
        return title
def only_title(name_file):
    """Extract the title from a beautiful soup title tag"""
    messy_title = str(get_title(filename=name_file))
    title_group = re.search(r"<title data-react-helmet=\"true\">(.*)</title", messy_title)
    title = title_group.group(1)
    return title

def get_date(filename="test.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the meta tags with content, then return the last one because it contains
    a url with the date in it. 
    """
    with open(filename, "r") as my_file:
        file_string = my_file.read()
        file_soup = BeautifulSoup(file_string, 'html.parser')
        #find all of the a tags with href attribute
        content = file_soup.select("meta[content]")
        date_url = content[-1]
        #use regular expression to get the date
        return date_url

def only_date(name_file):
    """Extract the date from a beautiful soup tag using regex
    the date is found in a meta content tag
    date is of the form YYYY/MM"""
    messy_date = str(get_date(name_file))
    date_group = re.search(r"<meta content=\"\/study\/general-conference\/(\d\d\d\d\/\d\d)", messy_date)
    date = date_group.group(1)
    return date

def create_key(filename):
    """The key of our dictionary will be the title and date of the talk
    This should be a unique combination for all talks
    This function gets the date and title of a talk given the html filename
    The key is of the form title:date"""
    date = only_date(filename)
    title = only_title(filename)
    key = title + ":" + date
    return key

directory='./churchofjesuschrist.org'
def create_topic_dictionary(directory):
    """Given a directory, parse through the topics in the directory
    then parse through each file in the folder to get the talks that
    are of each topic"""
    topics = os.listdir(directory)
    topic_dictionary = dict()
    for topic in topics:
        new_direct = directory + '/' + topic + '/'
        for talk in os.listdir(new_direct):
            talk_path = new_direct + talk
            if talk.endswith(".html"):
                key = create_key(talk_path)
                if key in topic_dictionary:
                    topic_dictionary[key].append(topic)
                else:
                    topic_dictionary[key] = list()
                    topic_dictionary[key].append(topic)
    return topic_dictionary

if __name__=="__main__":
	directory='./churchofjesuschrist.org'
	topic_dictionary = create_topic_dictionary(directory)
	data = pd.DataFrame.from_dict(topic_dictionary, orient='index')
	data.to_csv('topic_data.csv')
