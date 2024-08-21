import torch
from sentence_transformers import SentenceTransformer

# Load the embeddings once
prod2prod_embeddings = torch.load('path/to/main_model_combined_embeddings.pt')

# Load the model once
model_semantic_search = SentenceTransformer('multi-qa-mpnet-base-cos-v1')