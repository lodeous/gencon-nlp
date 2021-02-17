# utils.py

import string

def get_topic_code(topic_string: str) -> str:
    """Takes a string and returns it in topic code
    format (snake case) with punctuation removed.
    """
    remove_punc = topic_string.translate(str.maketrans('', '', string.punctuation))
    topic_code = str.lower(remove_punc).replace(' ', '_')
    return topic_code

def format_topic_code(topic_code: str) -> str:
    """Takes a topic code string and formats it as 
    human readable text.
    """
    return str.title(topic_code.replace('_', ' '))
