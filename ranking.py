"""
Workflow for the ranking file:
    funtctions:
        search_page_request(query): If the function is called, it simply calls find_closest_matches (from product-to-product import find_closest_matches)
        in_cart_request(product_id, user_id): This function will call user_info_retrieval (placeholder), then it'll check if user has at least 3 unqiue order_id against the user_id
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
        product_page_request()
"""

# Importing the required function from product-to-product
from product_to_product import find_closest_matches
from data_loader import products_with_ratings_aisle_department

def search_page_request(query):
    """
    Handles the request to search for products based on a query.

    Args:
        query (str): The search query input by the user.

    Returns:
        DataFrame: The DataFrame containing the closest matching products.
    """

    # Placeholder values for the other parameters
    threshold = 30
    rating_weight = 0.05
    products_needed = 40

    # Call the find_closest_matches function with the query
    return find_closest_matches(query, products_with_ratings_aisle_department, threshold=threshold, rating_weight=rating_weight, products_needed=products_needed)

def in_cart_request(product_id, user_id):
    """
    Handles the in-cart recommendations based on user history and product similarity.

    Args:
        product_id (int): The product ID for the current item in the cart.
        user_id (int): The user ID of the customer.

    Returns:
        List[int]: A list of recommended product IDs (at most 5) for the user.
    """

    # Placeholder: Retrieve user information
    user_info_df = user_info_retrieval(user_id)
    
    # Check if the user has at least 3 unique order_ids
    unique_orders = user_info_df['order_id'].nunique()
    
    recommended_products = set()

    if unique_orders >= 3:
        # Scenario: User has sufficient purchase history

        # Step 1: Call user_group_products function
        cluster_number = user_info_df['cluster_number'].iloc[0]  # Assuming cluster_number is retrieved from user_info_df
        user_group_recommendations = user_group_products(cluster_number, product_id)
        recommended_products.update(user_group_recommendations)

        # Step 2: Call frequently_bought_products function
        frequently_bought_recommendations = frequently_bought_products(product_id)
        recommended_products.update(frequently_bought_recommendations)

        # Step 3: Call parse_embeddings function to get product recommendations
        product_name = get_product_name(product_id)
        embedding_recommendations = parse_embeddings(product_name, top_n=5)
        recommended_products.update(embedding_recommendations)

        # Step 4: Call user_frequently_bought_products function
        user_frequent_recommendations = user_frequently_bought_products(user_id, product_id)
        recommended_products.update(user_frequent_recommendations)

        # Ensure we get at most 5 unique products and prioritize the recommendations
        final_recommendations = list(recommended_products)[:5]

        # Compensate if there are fewer than 5 items
        if len(final_recommendations) < 5:
            remaining_slots = 5 - len(final_recommendations)
            backup_recommendations = (
                list(user_frequent_recommendations) +
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
        # Scenario: User has limited purchase history, focus on frequently bought and embeddings
        frequently_bought_recommendations = frequently_bought_products(product_id)
        embedding_recommendations = parse_embeddings(product_name, top_n=5)

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

