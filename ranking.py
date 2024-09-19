"""
Workflow for the ranking file:
    funtctions:
        search_page_request(query): If the function is called:
                            It'll tale in take in the user query
                            it'll first seach using get_closest_matches
                            if the products are less than products_needed send the search query to get_products_from_embeddings
                            then take those searches and append them underneath
                            and at the end take the top items as per the products_needed number and return those
        product_page_request(product_id, user_id): This function will call user_info_retrieval (placeholder), then it'll check if user has at least 3 unqiue order_id against the user_id
                            If yes:
                                1. then we will call user_group_products function (placeholder), it'll take in the cluster number (from df returned by user_info_retrieval),
                                and product_id which is passed to in_cart_request function and will get a list of product_ids
                                2. then it'll call frequently_bought_products function, it'll take in product_id which is passed to in_cart_request function, it'll also return a list of product_ids
                                3. then it'll call parse_embeddings function by passing in the product_name, top_n=5; we will retrieve product_name through product_id from get product_name function (placeholder)
                                4. then it'll call user_frequently_bought_products (placeholder) and pass the product_id and will get a list of product_ids back.
                                5. we need to send back at most 5 product ids for suggestions, we will ideally 2 from user_frequently_bought_products, 1 from frequently_bought_products, 1 from user_group_products, and 1 from parse_embeddings
                                    we will also make sure no duplicated product ids are being sent and we will also make sure if no products are return from a certain function it should be compensated by othe rfunctiosn so we ar enot sending less than 5 items
                            If No: 
                                we will skip user_group_products and user_frequently_bought_products and use products from just frequently_bought_products and parse_embeddings, and ideally
                                use 3 from frequently_bought_products and 2 from parse_embeddings
        in_cart_request(): This function will call user_info_retrieval (placeholder), then it'll check if user has at least 3 unqiue order_id against the user_id
                            If yes:
                                1. then we will call user_group_products function (placeholder), it'll take in the cluster number (from df returned by user_info_retrieval),
                                and product_id which is passed to in_cart_request function and will get a list of product_ids
                                2. then it'll call frequently_bought_products function, it'll take in product_id which is passed to in_cart_request function, it'll also return a list of product_ids
                                3. ideally we will need to return a df with 3 products from user_group_productsand 2 from frequently_bought_products
                            If No:
                                Skip user_group_products function and just give top 5 from frequently_bought_products
        most_popular_items()
"""

# Importing the required function from product-to-product
from frequently_bought_together import get_frequently_bought_products, get_frequently_bought_user_based
from product_to_product import get_products_from_embeddings, all_in_one_search
from utils import user_info_retrieval, get_product_name, get_cluster_from_user_id, get_product_name_uncased
from data_loader import products_with_ratings_aisle_department
import pandas as pd


def search_page_request(query, threshold = 30):
    """
    Handles the request to search for products based on a query.

    Args:
        query (str): The search query input by the user.

    Returns:
        DataFrame: The DataFrame containing the closest matching products.
    """
    # Placeholder values for the other parameter
    rating_weight = 0.05
    products_needed = 40

    # # Step 1: Get closest matches using the query
    # closest_matches_df = get_closest_matches(query, products_with_ratings_aisle_department, threshold=threshold, rating_weight=rating_weight, products_needed=products_needed)

    # # Step 2: Check if the number of products found is less than needed
    # if len(closest_matches_df) < products_needed:
    #     # Fetch additional products using embeddings
    #     additional_products_df = get_products_from_embeddings(query, products_with_ratings_aisle_department, rating_weight=rating_weight, top_n=products_needed)

    #     # Append additional products to the closest matches
    #     combined_df = closest_matches_df.append(additional_products_df, ignore_index=True)
    #     del closest_matches_df  # Delete the original closest matches DataFrame
    # else:
    #     combined_df = closest_matches_df

    # # Step 3: Ensure no duplicates and limit to products_needed using indexing
    # combined_df = combined_df.drop_duplicates()

    # # Extract the top 'products_needed' using index
    # top_products_df = combined_df.iloc[:products_needed].reset_index(drop=True)

    # # Delete the combined DataFrame
    # del combined_df
    recommended_products_semantic = all_in_one_search(query, products_with_ratings_aisle_department, top_n=10)
    top_products_df = recommended_products_semantic.head(3)
    # Return the new DataFrame with top 'products_needed' products
    return top_products_df


def product_page_request(product_id, user_id):
    """
    Handles the in-cart recommendations based on user history and product similarity.

    Args:
        product_id (int): The product ID for the current item in the cart.
        user_id (int): The user ID of the customer.

    Returns:
        List[int]: A list of recommended product IDs (at most 5) for the user.
    """
    # Convert product_id and user_id to integers immediately
    try:
        product_id = int(product_id)
        user_id = int(user_id)
    except ValueError:
        # Handle cases where conversion fails
        print(f"Invalid input: product_id={product_id}, user_id={user_id} must be integers.")
        return None
    
    # Placeholder: Retrieve user information
    user_info_df = user_info_retrieval(user_id)
    # print(f"User Info:\n{user_info_df}")
    
    # Check if the user has at least 3 unique order_ids
    unique_orders = user_info_df['order_id'].nunique()
    
    recommended_products = set()
    product_name = get_product_name_uncased(product_id)
    print(f"Product: {product_name}")

    if unique_orders >= 3:
        print("user cluster available")
        # Scenario: User has sufficient purchase history

        # Step 1: Call user_group_products function
        # cluster_number = user_info_df['cluster_number'].iloc[0]  # Assuming cluster_number is retrieved from user_info_df
        cluster_number = get_cluster_from_user_id(user_id)
        user_group_recommendations = get_frequently_bought_user_based(cluster_number, product_id)
        recommended_products.update(user_group_recommendations)
        print(f"recommendations from user_group_recommendations: {user_group_recommendations}")


        # Step 2: Call frequently_bought_products function
        if not isinstance(product_id, list):
            product_id_list = [product_id]
        frequently_bought_recommendations = get_frequently_bought_products(product_id_list)
        recommended_products.update(frequently_bought_recommendations)
        print(f"recommendations from frequently_bought_recommendations: {frequently_bought_recommendations}")


        # Step 3: Call parse_embeddings function to get product recommendations
        
        embedding_recommendations = get_products_from_embeddings(product_name, products_with_ratings_aisle_department, top_n=5)
        recommended_products.update(embedding_recommendations)
        print(f"recommendations from embeddings: {embedding_recommendations}")

        # Step 4: Call user_frequently_bought_products function
        # user_frequent_recommendations = user_frequently_bought_products(user_id, product_id)
        # recommended_products.update(user_frequent_recommendations)

        # Ensure we get at most 5 unique products and prioritize the recommendations
        final_recommendations = list(recommended_products)[:5]

        # Compensate if there are fewer than 5 items
        if len(final_recommendations) < 5:
            remaining_slots = 5 - len(final_recommendations)
            backup_recommendations = (
                # list(user_frequent_recommendations) +
                list(frequently_bought_recommendations) +
                list(user_group_recommendations) +
                list(embedding_recommendations)
            )
            for product in backup_recommendations:
                if len(final_recommendations) >= 5:
                    break
                if product not in final_recommendations:
                    final_recommendations.append(product)

    else:
        print("user cluster not available")

        # Scenario: User has limited purchase history, focus on frequently bought and embeddings
        if not isinstance(product_id, list):
            product_id_list = [product_id]
        frequently_bought_recommendations = get_frequently_bought_products(product_id_list)
        embedding_recommendations = get_products_from_embeddings(product_name, products_with_ratings_aisle_department,top_n=5)

        final_recommendations = list(frequently_bought_recommendations)[:3] + list(embedding_recommendations)[:2]

        # Ensure no duplicates and compensate if fewer than 5
        final_recommendations = list(set(final_recommendations))
        if len(final_recommendations) < 5:
            remaining_slots = 5 - len(final_recommendations)
            backup_recommendations = list(frequently_bought_recommendations) + list(embedding_recommendations)
            for product in backup_recommendations:
                if len(final_recommendations) >= 5:
                    break
                if product not in final_recommendations:
                    final_recommendations.append(product)

    # Ensure at most 5 unique recommendations
    return final_recommendations[:5]

def in_cart_request(product_id, user_id):
    """
    Handles the in-cart recommendations based on user history.

    Args:
        product_id (int): The product ID for the current item in the cart.
        user_id (int): The user ID of the customer.

    Returns:
        DataFrame: A DataFrame containing the recommended product IDs.
    """

    try:
        product_id = int(product_id)
        user_id = int(user_id)
    except ValueError:
        # Handle cases where conversion fails
        print(f"Invalid input: product_id={product_id}, user_id={user_id} must be integers.")
        return None
    
    # Retrieve user information
    user_info_df = user_info_retrieval(user_id)

    # Check if the user has at least 3 unique order_ids
    unique_orders = user_info_df['order_id'].nunique() if not user_info_df.empty else 0

    if unique_orders >= 3:
        print("user in cluster")
        # User has sufficient purchase history

        # Step 1: Call user_group_products function
        # cluster_number = user_info_df['cluster_number'].iloc[0]  # Assuming cluster_number is retrieved from user_info_df
        cluster_number = get_cluster_from_user_id(user_id)
        user_group_recommendations = get_frequently_bought_user_based(cluster_number, product_id)
        
        # Step 2: Call frequently_bought_products function
        if not isinstance(product_id, list):
            product_id_list = [product_id]
        frequently_bought_recommendations = get_frequently_bought_products(product_id_list)

        # Step 3: Combine results to prepare final recommendations using indexing
        recommended_products = user_group_recommendations[:3]  # Select first 3 products from user_group_recommendations
        recommended_products += frequently_bought_recommendations[:2]  # Select first 2 products from frequently_bought_recommendations

    else:
        print("user not in cluster")
        # User has limited purchase history
        # Get top 5 products from frequently_bought_products using indexing
        if not isinstance(product_id, list):
            product_id_list = [product_id]
        recommended_products = get_frequently_bought_products(product_id_list)[:5]

    return recommended_products

