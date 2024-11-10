import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download NLTK data files if not already present
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

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


# Preprocess text for recommendation
df['Preprocessed_Plot'] = df['authors'].apply(preprocess_text)

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Preprocessed_Plot'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Recommendation function
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

# App Title and Header
st.markdown("<h1 style='text-align: center; color: #4B0082;'>📚 Fiction Book Recommendation System 📚</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #808080;'>Find your next great read based on books you love!</p>",
            unsafe_allow_html=True)
st.divider()

# Book Title Input
st.markdown("### 📖 Enter a book title to get recommendations:")
title = st.text_input("")

# Recommendation Button
if st.button("🔍 Recommend Books"):
    if title:
        with st.spinner('Finding recommendations...'):
            time.sleep(1)
            recommendations = recommend_similar_books(title)

            if recommendations:
                st.markdown("### 📚 Recommended Books")
                for book in recommendations:
                    st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                            <h4 style="color: #4B0082;">{book['title']}</h4>
                            <p><strong>Author:</strong> {book['author']}</p>
                            <p><strong>Average Rating:</strong> ⭐ {book['rating']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try a different title.")
    else:
        st.warning("Please enter a title.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Sylvester P.K Pro</p>", unsafe_allow_html=True)
