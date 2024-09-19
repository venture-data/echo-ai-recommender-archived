from data_loader import group_association_rules_dic, rules_freq_bought
from utils import get_product_name, get_product_name_uncased

# Function to get recommendations based on association rules
def get_frequently_bought_products(cart_items, top_n=5):
    # First, get unique product ids from the cart items
    unique_cart_items = list(set(cart_items))

    # Replace product IDs with product names, converting product_id to an integer
    cart_items_names = []
    for product_id in unique_cart_items:
        try:
            # Convert product_id to integer
            product_id_int = int(product_id)
            product_name = get_product_name_uncased(product_id_int)
            if product_name is not None:
                cart_items_names.append(product_name)
        except ValueError:
            print(f"Invalid product_id: {product_id}. Skipping.")
    
    # Debugging print statement to show the processed cart items
    print(f'cart items: {cart_items_names}')
    
    # Convert cart items to a frozenset to match the antecedents in association rules
    cart_items_set = frozenset(cart_items_names)
    
    # Find rules where the antecedents (items in the cart) are a subset of the rule antecedents
    recommendations = rules_freq_bought[rules_freq_bought['antecedents'].apply(lambda x: cart_items_set.issubset(x))]
    print(f"recommendations: {recommendations}")
    
    # Extract only the product names from the 'consequents'
    top_recommendations = recommendations['consequents'].explode().unique().tolist()
    
    # Return the top N product names (limit to top_n)
    return top_recommendations[:top_n]


def get_frequently_bought_user_based(cluster_id, product_id, num_products=5):
    # Retrieve the association rules for the given cluster
    df = group_association_rules_dic[cluster_id]
    
    # Find the rules where the product is either item_A or item_B
    df = df[(df['item_A'] == product_id) | (df['item_B'] == product_id)][[
        'product_name_A', 'product_name_B', 'item_A', 'item_B', 'confAtoB', 'lift'
    ]]
    
    # Sort by lift to get the strongest associations
    df = df.sort_values('lift', ascending=False)
    
    # Collect the frequently bought together products
    frequently_bought_together = df['product_name_A'].tolist() + df['product_name_B'].tolist()
    
    # Ensure the original product is excluded from the result and remove duplicates
    frequently_bought_together = [x for x in set(frequently_bought_together) if x != get_product_name(product_id)]
    
    # Return the top N frequently bought together products
    return frequently_bought_together[:num_products]

