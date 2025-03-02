# Aeka-Project-02
#username: Romit Pal
Song Lyrics Similarity Search
This project uses a pre-trained DistilBERT model to compute embeddings for song lyrics and find the most similar songs based on a user-provided snippet. It leverages cosine similarity to compare embeddings and returns matching songs from a dataset.

Prerequisites
Python: Version 3.8 or higher
Operating System: Windows, macOS, or Linux
Dataset: A CSV file named spotify_songs.csv with at least the columns artist, song, and text (containing lyrics or text data).
Installation
Follow these steps to set up the environment and install required dependencies.

1. Clone or Download the Project
If this is part of a repository, clone it:

2. Set Up a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment to manage dependencies:

pip install torch transformers pandas scikit-learn numpy
torch: PyTorch library for tensor operations.
transformers: Hugging Face library for DistilBERT model and tokenizer.
pandas: Data manipulation and CSV handling.
scikit-learn: Cosine similarity computation.
numpy: Numerical operations with arrays.
4. Prepare the Dataset
Place the spotify_songs.csv file in a Dataset subdirectory relative to the script:
<project-directory>/Dataset/spotify_songs.csv
The CSV must contain at least three columns: artist, song, and text.
Example structure:
artist,song,text
"The Eagles","Hotel California","On a dark desert highway, cool wind in my hair..."
Execution:
5.Run the python code: python main.py

Input: A snippet of lyrics
Output: The song name followed by the song writer name based onn the given dataset
