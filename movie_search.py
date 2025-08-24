import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#loading the dataset once
movies_df=pd.read_csv("movies.csv")

#load model once 
model=SentenceTransformer('all-MiniLM-L6-v2')

#pre compute embeddings for all the plots 
movie_embeddings=model.encode(movies_df['plot'].tolist(), convert_to_tensor=False)

def search_movies(query, top_n=5):
    """
    Search for movies similar to a query using semantic similarity.

    Args:
        query (str): The search query (e.g., "spy thriller in Paris").
        top_n (int): Number of top results to return.

    Returns:
        pd.DataFrame: Top N most similar movies with title, plot, and similarity score.
    """

    #encode query
    query_embedding = model.encode([query], convert_to_tensor=False)

    #compute cosine similarity
    similarities = cosine_similarity(query_embedding, movie_embeddings)[0]

    #attach similarity scores to dataframe
    movies_df_copy = movies_df.copy()
    movies_df_copy['similarity'] = similarities

    #sort by similarity and return top_n
    return movies_df_copy.sort_values(by='similarity', ascending=False).head(top_n)[['title', 'plot', 'similarity']]
