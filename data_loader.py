import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

prod2prod_embeddings = torch.load('/content/drive/MyDrive/recommender_system/Ammar Embeddings/semantic_model_combined_embeddings.pt')
model_semantic_search = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

products_with_ratings_aisle_department = pd.read_csv('/content/drive/MyDrive/recommender_system/Embeddings/product_random_normalized_ratings.csv')
# users_with_orders = pd.read_csv('path_to_your_csv_file.csv')
