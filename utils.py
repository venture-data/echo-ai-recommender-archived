from fuzzywuzzy import process
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
# from data_loader import user_purchase_data

user_purchase_data = []


def extract_product_name(query, csv_df, threshold=90):
    product_names = csv_df['product_name'].unique()
    match, score = process.extractOne(query, product_names, scorer=fuzz.token_set_ratio)
    if score >= threshold:
        return match
    return None

def exact_match_product_name(query, csv_df):
    product_names = csv_df['product_name'].unique()
    query_lower = query.lower().strip()
    exact_matches = [name for name in product_names if name.lower() == query_lower]
    return exact_matches[0] if exact_matches else None

def extract_aisle(query, csv_df):
    # List of known aisles
    aisles = csv_df['aisle'].unique()
    matches = [aisle for aisle in aisles if aisle.lower() in query.lower()]
    return matches[0] if matches else None

def extract_department(query, csv_df):
    # List of known departments
    departments = csv_df['department'].unique()
    matches = [dept for dept in departments if dept.lower() in query.lower()]
    return matches[0] if matches else None

def get_correct_spelling(query, choices, threshold=80):
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

def user_info_retrieval(user_id):
    """
    Retrieves user information from the user purchase data DataFrame.

    Args:
        user_id (int): The user ID to filter the data.

    Returns:
        DataFrame: A DataFrame containing all rows where the user ID matches.
    """
    # Assuming user_purchase_data is a DataFrame loaded from another file
    filtered_df = user_purchase_data[user_purchase_data['user_id'] == user_id]
    return filtered_df