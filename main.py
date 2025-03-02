import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

torch.set_num_threads(1)  
file_path = './Dataset/spotify_songs.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file not found at {file_path}")

df = pd.read_csv(file_path)
if 'text' not in df.columns:
    raise ValueError("Dataset must contain a 'text' column")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")


def get_embedding(text):
    """Generate embedding for a given text using DistilBERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

df = df.head(100) 
df['embedding'] = df['text'].astype(str).map(lambda x: get_embedding(x))


def find_song_lyrics_snippet(snippet, top_n=1):
    """Find the most similar song based on a text snippet using deep learning embeddings."""
    print(f"Processing snippet: {snippet}")
    snippet_embedding = get_embedding(snippet)
    embeddings_matrix = np.vstack(df['embedding'].values)
    similarities = cosine_similarity([snippet_embedding], embeddings_matrix)[0]
    
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    result = df.iloc[top_indices][['artist', 'song', 'text']]
    print("Top matches found:")
    print(result)
    return result


print("Running test query...")
user_snippet = input("Enter a song lyrics snippet to search for: ")
find_song_lyrics_snippet(user_snippet)