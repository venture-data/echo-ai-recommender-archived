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
