import torch
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

prod2prod_embeddings = torch.load('/content/drive/MyDrive/recommender_system/Ammar Embeddings/semantic_model_combined_embeddings.pt')

product_name_embedding = torch.load("")
aisle_embedding = torch.load("")
department_embedding = torch.load("")

model_semantic_search = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

products_with_ratings_aisle_department = pd.read_csv('/content/drive/MyDrive/recommender_system/Embeddings/product_random_normalized_ratings.csv')
# users_with_orders = pd.read_csv('path_to_your_csv_file.csv')

with open('/content/drive/MyDrive/recommender_system/Embeddings/group_association_rules_dic.pkl', 'rb') as f:
    group_association_rules_dic = pickle.load(f)

# Load frequent itemsets
with open('/content/drive/MyDrive/recommender_system/Embeddings/frequent_itemsets.pkl', 'rb') as f:
    frequent_itemsets = pickle.load(f)

# Load association rules
with open('/content/drive/MyDrive/recommender_system/Embeddings/association_rules.pkl', 'rb') as f:
    rules = pickle.load(f)