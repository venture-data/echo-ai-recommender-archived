from data_loader import group_association_rules_dic, rules_freq_bought

# Function to get recommendations based on association rules
def get_frequently_bought_products(cart_items, top_n=3):
    # Convert cart items to a frozenset to match the antecedents in association rules
    cart_items_set = frozenset(cart_items)
    # Find rules where the antecedents (items in the cart) are a subset of the rule antecedents
    recommendations = rules_freq_bought[rules_freq_bought['antecedents'].apply(lambda x: cart_items_set.issubset(x))]
    # Sort the recommendations by confidence and lift, and return top N
    top_recommendations = recommendations[['consequents', 'confidence', 'lift']].sort_values(by='confidence', ascending=False).head(top_n)
    return top_recommendations

def get_frequently_bought_user_based(cluster_id, product_id, num_products=5):
    # Retrieve the association rules for the given cluster
    df = group_association_rules_dic[cluster_id]
    
    # Find the rules where the product is either item_A or item_B
    df = df[(df['item_A'] == product_id) | (df['item_B'] == product_id)][[
        'product_name_A', 'item_A', 'product_name_B', 'item_B', 'confAtoB', 'lift'
    ]]
    
    # Sort by lift to get the strongest associations
    df = df.sort_values('lift', ascending=False)
    
    # Select top N products based on the lift
    df = df.head(n=num_products)
    
    # Collect the frequently bought together products
    frequently_bought_together = df['product_name_A'].values.tolist()
    frequently_bought_together += df['product_name_B'].values.tolist()
    
    # Ensure the original product is excluded from the result
    frequently_bought_together = [x for x in frequently_bought_together if x != product_id]
    
    return frequently_bought_together
