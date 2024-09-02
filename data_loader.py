import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Load the DataFrame once
# Load the embeddings once
prod2prod_embeddings = torch.load('path/to/main_model_combined_embeddings.pt')
# Load the model once
model_semantic_search = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

products_with_ratings_aisle_department = pd.read_csv('path_to_your_csv_file.csv')
users_with_orders = pd.read_csv('path_to_your_csv_file.csv')