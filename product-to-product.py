import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from main import prod2prod_embeddings, model_semantic_search


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

def parse_embeddings(csv_df, query, rating_weight=0.05, top_n=100):
    # Split the query into components (assuming the query contains all three components)
    query_product_name = query
    query_aisle = query
    query_department = query

    # Generate the embeddings for each component and move to GPU
    query_product_name_embedding = model_semantic_search.encode(query_product_name, convert_to_tensor=True).to('cuda')
    query_aisle_embedding = model_semantic_search.encode(query_aisle, convert_to_tensor=True).to('cuda')
    query_department_embedding = model_semantic_search.encode(query_department, convert_to_tensor=True).to('cuda')
    
    # Concatenate the embeddings to form the combined query embedding
    query_embedding = torch.cat([query_product_name_embedding, query_aisle_embedding, query_department_embedding], dim=-1).unsqueeze(0)

    # Ensure query_embedding has the correct dimensions
    assert query_embedding.shape[1] == prod2prod_embeddings.shape[1], "Query embedding dimension mismatch."

    # Compute cosine similarity between the query embedding and product embeddings
    cosine_sim_query = util.pytorch_cos_sim(query_embedding, prod2prod_embeddings).cpu().numpy()[0]

    # Normalize similarity scores to 0-1 range
    normalized_similarity = (cosine_sim_query - cosine_sim_query.min()) / (cosine_sim_query.max() - cosine_sim_query.min())

    # Apply a boosting function to the normalized ratings
    boosted_ratings = np.power(csv_df['normalized_ratings'].values, 2)

    # Combine similarity scores with boosted ratings
    combined_scores = (1 - rating_weight) * normalized_similarity + rating_weight * boosted_ratings

    # Get the similarity scores for the query
    sim_scores = list(enumerate(combined_scores))

    # Sort the products based on combined scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top N similar products
    sim_scores = sim_scores[:top_n]

    # Get the product IDs of the top N similar products
    product_indices = [i[0] for i in sim_scores]

    return csv_df.iloc[product_indices]