import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


"""
    Work Flow:
        1. A person Selects a Product
            - Embeddings are used to suggest similar products in terms of name/aisle/department (give weights to ratings)
        2. A user Searches for a Product
            - We need to show exact matches of the product (give weights to ratings)
            - Get names of products from DB directly
            - Calculate Levenstine distance
            - Keep all the products that are close by
                    (TEST) :
                                Chips Ahoy!
            - Then use fuzzy to correct spellings
            - Then use embeddings to get the products, and give weights to ratings
"""

def parse_embeddings(csv, query, rating_weight=0.05, top_n=100):
    # Split the query into components (assuming the query contains all three components)
    query_product_name = query
    query_aisle = query
    query_department = query

    # Generate the embeddings for each component and move to GPU
    query_product_name_embedding = prod2prod_model.encode(query_product_name, convert_to_tensor=True).to('cuda')
    query_aisle_embedding = prod2prod_model.encode(query_aisle, convert_to_tensor=True).to('cuda')
    query_department_embedding = prod2prod_model.encode(query_department, convert_to_tensor=True).to('cuda')

