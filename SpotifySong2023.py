import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess data and compute clustering
def preprocess_and_cluster(df, n_clusters=5):
    features_for_clustering = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'instrumentalness_%']
    df_cleaned = df.dropna(subset=features_for_clustering + ['track_name', 'artist(s)_name'])

    # Scale features
    scaler = MinMaxScaler()
    df_cleaned[features_for_clustering] = scaler.fit_transform(df_cleaned[features_for_clustering])

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cleaned['cluster'] = kmeans.fit_predict(df_cleaned[features_for_clustering])

    return df_cleaned, kmeans

# Find similar songs
def find_similar_songs(df, song_title, artist_name):
    features_for_clustering = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'instrumentalness_%']

    # Find the input song
    song = df[(df['track_name'].str.contains(song_title, case=False, na=False)) &
              (df['artist(s)_name'].str.contains(artist_name, case=False, na=False))]
    if song.empty:
        return None, None

    song_features = song.iloc[0][features_for_clustering].values
    input_cluster = song.iloc[0]['Cluster']

    # Filter songs in the same cluster and compute Euclidean distance
    cluster_songs = df[df['Cluster'] == input_cluster]
    cluster_songs['distance'] = cluster_songs[features_for_clustering].apply(lambda x: euclidean(song_features, x), axis=1)
    similar_songs = cluster_songs.sort_values(by='distance').head(11)  # Include input song

    return song.iloc[0], similar_songs[1:]  # Exclude input song

# Streamlit App
st.title("Song Similarity Finder with Clustering")
st.write("Find songs with similar characteristics and identify their cluster.")

# File path for dataset
data_file_path = 'clustering_results.csv'
data = load_data(data_file_path)
st.write("Dataset loaded successfully!")

# Perform clustering
clustered_data = data  # Data already clustered in the provided file

# User input
song_title = st.text_input("Enter a song title:")
artist_name = st.text_input("Enter the artist's name:")

if song_title and artist_name:
    song, similar_songs = find_similar_songs(clustered_data, song_title, artist_name)
    if song is not None:
        st.write("### Input Song Characteristics and Cluster:")
        st.write(song[['track_name', 'artist(s)_name', 'danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'instrumentalness_%', 'Cluster', 'Cluster Label']])

        st.write("### Similar Songs in the Same Cluster:")
        st.write(similar_songs[['track_name', 'artist(s)_name', 'danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'instrumentalness_%', 'Cluster', 'Cluster Label']])
    else:
        st.write("Song not found in the dataset. Please try another title or artist.")
