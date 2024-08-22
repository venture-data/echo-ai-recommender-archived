import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix, vstack


# Function to get recommendations based on association rules
def get_frequently_bought_products(cart_items, rules, top_n=3):
    cart_items_set = frozenset(cart_items)
    recommendations = rules[rules['antecedents'].apply(lambda x: cart_items_set.issubset(x))]
    top_recommendations = recommendations[['consequents', 'confidence', 'lift']].sort_values(by='confidence', ascending=False).head(top_n)
    return top_recommendations