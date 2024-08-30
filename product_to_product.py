import numpy as np
import torch
import Levenshtein
from sentence_transformers import util
import pandas as pd
from main import prod2prod_embeddings, model_semantic_search


"""
    Work Flow:
        1. A person Selects a Product /
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


def get_closest_matches(query, csv_df, threshold=30, rating_weight=0.05, products_needed=40):
    """
    Calculate Levenshtein distance for each product name and return those within the threshold.
    Ratings are only used for sorting and not for filtering.
    """

    # Step 1: Exact Match Search
    filtered_df = csv_df[csv_df['product_name'].str.contains(query, case=False, na=False)]

    if not filtered_df.empty:
        filtered_products = []
        for index, row in filtered_df.iterrows():
            product_name = row['product_name']
            distance = Levenshtein.distance(query.lower(), product_name.lower())

            # Boost score for exact substring matches
            if query.lower() in product_name.lower():
                distance = max(1, distance - 40)  # Reduce the distance for substring matches to prioritize them

            # Filter products based solely on distance
            if distance <= threshold:
                filtered_products.append((index, distance))

        # Sort the filtered products by weighted score (distance + rating)
        sorted_products = sorted(
            filtered_products, 
            key=lambda x: (1 - rating_weight) * (1 - (x[1] / threshold)) + rating_weight * csv_df.loc[x[0], 'normalized_ratings'], 
            reverse=True
        )

        # Get the indices of the top products after sorting
        match_indices = [match[0] for match in sorted_products]
        filtered_results = csv_df.iloc[match_indices]

        # If the number of filtered products is sufficient, return them
        if len(filtered_results) >= products_needed:
            return filtered_results.head(products_needed)
        else:
            # Return the available filtered results
            return filtered_results.head(products_needed)

    # If no direct matches are found, return an empty DataFrame
    return pd.DataFrame()


def get_products_from_embeddings(query, csv_df, rating_weight=0.05, top_n=100):
    """_summary_

    Args:
        csv_df (dataframe): its the data frame which will be used to return the products
        query (string): it's the search query
        rating_weight (float, optional): how much priority does the rating has to the search. Defaults to 0.05.
        top_n (int, optional): number of products to return. Defaults to 100.

    Returns:
        dataframe: the products with sorting
    """
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