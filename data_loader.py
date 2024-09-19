import torch
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load file paths from .env
prod2prod_embeddings_path = os.getenv('PROD2PROD_EMBEDDINGS')
product_name_embedding_path = os.getenv('PRODUCT_NAME_EMBEDDING')
aisle_embedding_path = os.getenv('AISLE_EMBEDDING')
department_embedding_path = os.getenv('DEPARTMENT_EMBEDDING')

products_with_ratings_path = os.getenv('PRODUCTS_WITH_RATINGS_DF')
user_id_cluster_path = os.getenv('USER_ID_WITH_CLUSTER_DF')
orders_df_path = os.getenv('ORDERS_DF')
products_not_cased_path = os.getenv('PRODUCTS_WIHTOUT_CASE_DF')

group_association_rules_path = os.getenv('GROUP_ASSOCIATION_RULES')
rules_freq_bought_path = os.getenv('RULES_FREQ_BOUGHT')

# Load data using the paths
prod2prod_embeddings = torch.load(prod2prod_embeddings_path, weights_only=True)
product_name_embedding = torch.load(product_name_embedding_path, weights_only=True)
aisle_embedding = torch.load(aisle_embedding_path, weights_only=True)
department_embedding = torch.load(department_embedding_path, weights_only=True)

model_semantic_search = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

# dataframe loading
products_with_ratings_aisle_department = pd.read_csv(products_with_ratings_path)
user_id_cluster = pd.read_csv(user_id_cluster_path)
orders_df = pd.read_csv(orders_df_path)
products_not_cased = pd.read_csv(products_not_cased_path)

with open(group_association_rules_path, 'rb') as f:
    group_association_rules_dic = pickle.load(f)

with open(rules_freq_bought_path, 'rb') as f:
    rules_freq_bought = pickle.load(f)

