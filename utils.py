from fuzzywuzzy import process
from nltk.corpus import wordnet

def correct_spelling(query, choices, threshold=80):
    """
    Corrects the spelling of the given query by matching it with the closest valid words.

    Args:
        query (str): The original search query.
        choices (list): A list of valid words (e.g., product names, dictionary words).
        threshold (int): The similarity threshold to consider a match (default: 80).

    Returns:
        str: The corrected query if a match is found; otherwise, the original query.
    """
    best_match, score = process.extractOne(query, choices)
    
    if score >= threshold:
        return best_match
    else:
        return query

def get_synonyms(query):
    """
    Generate synonyms for words in the query using WordNet.

    Args:
        query (str): The original search query.

    Returns:
        list: A list of synonyms for the words in the query.
    """
    synonyms = []
    for word in query.lower().split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
    return list(set(synonyms))  # Return unique synonyms