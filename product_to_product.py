import numpy as np
import torch
import Levenshtein
from sentence_transformers import util
import pandas as pd
from data_loader import prod2prod_embeddings, model_semantic_search
from utils import extract_product_name, exact_match_product_name, extract_aisle, extract_department
from data_loader import product_name_embedding, aisle_embedding, department_embedding

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
                                fruit yogurt
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

        # Sort the filtered products by Levenshtein distance first, then by weighted score
        sorted_products = sorted(
            filtered_products,
            key=lambda x: (
                x[1],  # Sort primarily by Levenshtein distance (lower is better)
                -(1 - rating_weight) * (1 - (x[1] / threshold)) + rating_weight * csv_df.loc[x[0], 'normalized_ratings']
            )
        )

        # Get the indices of the top products after sorting
        match_indices = [match[0] for match in sorted_products]
        filtered_results = csv_df.iloc[match_indices]

        # If the number of filtered products is sufficient, return them
        if len(filtered_results) >= products_needed:
            return filtered_results.iloc[:products_needed]  # Use indexing to get the top products_needed
        else:
            # Return the available filtered results
            return filtered_results.iloc[:products_needed]  # Return all available results up to products_needed

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

def all_in_one_search(query, model_semantic_search, prod2prod_embeddings, csv_df, rating_weight=0.05, top_n=100):
    # Check for exact product name match
    exact_match = exact_match_product_name(query, csv_df)
    if exact_match:
        # Return the exact match with highest priority
        exact_match_df = csv_df[csv_df['product_name'] == exact_match].copy()
        exact_match_df['similarity_score'] = 1.0  # Perfect match
        return exact_match_df
    
    # Proceed with fuzzy matching if no exact match is found
    query_product_name = extract_product_name(query, csv_df)
    query_aisle = extract_aisle(query, csv_df)
    query_department = extract_department(query, csv_df)
    
    # Initialize weights
    weights_dict = {
        'product_name': 0.7,
        'aisle': 0.2,
        'department': 0.1
    }
    
    # Collect the components present in the query
    present_components = []
    if query_product_name:
        present_components.append('product_name')
    if query_aisle:
        present_components.append('aisle')
    if query_department:
        present_components.append('department')
    
    # If no components are found, default to using the entire query as product name
    if not present_components:
        query_product_name = query  # Use the entire query
        present_components.append('product_name')
    
    # Adjust weights to prioritize product name when present
    if 'product_name' in present_components:
        # Increase weight for product_name
        weights_dict['product_name'] = 0.9
        remaining_weight = 0.1
        other_components = [comp for comp in present_components if comp != 'product_name']
        total_other_weight = sum(weights_dict[comp] for comp in other_components)
        for comp in other_components:
            weights_dict[comp] = (weights_dict[comp] / total_other_weight) * remaining_weight if total_other_weight else 0
    
    # Normalize the weights for the present components
    total_weight = sum(weights_dict[comp] for comp in present_components)
    normalized_weights = {comp: weights_dict[comp] / total_weight for comp in present_components}
    
    embeddings = []
    weights = []
    
    # Generate embeddings only for the components present
    if query_product_name:
        product_name_embedding = model_semantic_search.encode(query_product_name, convert_to_tensor=True).to('cuda')
        embeddings.append(product_name_embedding)
        weights.append(normalized_weights['product_name'])
    if query_aisle:
        aisle_embedding = model_semantic_search.encode(query_aisle, convert_to_tensor=True).to('cuda')
        embeddings.append(aisle_embedding)
        weights.append(normalized_weights.get('aisle', 0))
    if query_department:
        department_embedding = model_semantic_search.encode(query_department, convert_to_tensor=True).to('cuda')
        embeddings.append(department_embedding)
        weights.append(normalized_weights.get('department', 0))
    
    # Weighted averaging of embeddings
    query_embedding = torch.stack(embeddings)
    weights_tensor = torch.tensor(weights).unsqueeze(1).to('cuda')  # Shape: (num_components, 1)
    query_embedding = torch.sum(query_embedding * weights_tensor, dim=0)  # Weighted sum
    query_embedding = query_embedding / sum(weights)  # Normalize by sum of weights
    query_embedding = query_embedding.unsqueeze(0)  # Add batch dimension
    
    # Ensure dimensions match
    assert query_embedding.shape[1] == prod2prod_embeddings.shape[1], "Query embedding dimension mismatch."
    
    # Compute cosine similarity
    cosine_sim_query = util.pytorch_cos_sim(query_embedding, prod2prod_embeddings).cpu().numpy()[0]
    
    # Apply boosted ratings
    boosted_ratings = np.power(csv_df['normalized_ratings'].values, 2)
    
    # Combine scores without normalization to preserve exact match significance
    combined_scores = (1 - rating_weight) * cosine_sim_query + rating_weight * boosted_ratings
    
    # If we had a fuzzy match for product name, boost its score
    if query_product_name:
        fuzzy_match_indices = csv_df[csv_df['product_name'] == query_product_name].index
        for idx in fuzzy_match_indices:
            combined_scores[idx] += combined_scores.max() * 0.1  # Boost by 10% of the max score
    
    # Sort and select top N products
    sim_scores = list(enumerate(combined_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    product_indices = [i[0] for i in sim_scores]
    top_products_df = csv_df.iloc[product_indices].copy()
    
    # Add similarity score to the dataframe
    top_products_df['similarity_score'] = [combined_scores[i[0]] for i in sim_scores]
    
    return top_products_df
