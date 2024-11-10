import streamlit as st
import pandas as pd
import time  # Import time module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the data
df = pd.read_csv('books_data.csv')


# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


# Apply preprocessing to the 'plot' or 'description' column (if you have one, for example 'plot')
df['Preprocessed_Plot'] = df['authors'].apply(preprocess_text)

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Preprocessed_Plot'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Recommendation function with additional details
def recommend_similar_books(title, cosine_sim=cosine_sim):
    if title not in df['title'].values:
        return ["Book title not found. Please try another title."]

    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]

    recommended_books = []

    for index in book_indices:
        book_details = {
            'title': df['title'].iloc[index],
            'author': df['authors'].iloc[index],
            'rating': df['average_rating'].iloc[index]
        }
        recommended_books.append(book_details)

    return recommended_books


# Streamlit App Configuration
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

# App Title
st.title("Global Book Recommendation System")

# Book Title Input
title = st.text_input("Enter a book title to get recommendations:")

# Recommend Button and Display Results with Spinner
if st.button("Recommend"):
    if title:
        with st.spinner('Finding recommendation...'):
            time.sleep(2)  # Adds a delay to ensure spinner visibility
            recommendations = recommend_similar_books(title)

            if recommendations:
                st.write("### Recommended Books:")
                for book in recommendations:
                    st.write(f"**Title**: {book['title']}")
                    st.write(f"**Author(s)**: {book['author']}")
                    st.write(f"**Average Rating**: {book['rating']}")
                    st.write("___")  # Adding a horizontal line to separate recommendations
            else:
                st.write("No recommendations found. Try a different title.")
    else:
        st.write("Please enter a title.")
