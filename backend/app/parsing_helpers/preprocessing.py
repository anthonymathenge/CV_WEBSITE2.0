# resume_parser/preprocessing.py
import json
import re
import nltk
from app.config.models import nlp, model, rake


# Download stopwords if not already
nltk.download("stopwords")
from nltk.corpus import stopwords

# Create stopword set (nltk + spacy combined)
stop_words = set(stopwords.words("english")).union(nlp.Defaults.stop_words)

# Degree normalization dictionary
with open("degree_map.json", "r", encoding="utf-8") as f:
    degree_map = json.load(f)

# Load skills_map from JSON
with open("skills_map.json", "r", encoding="utf-8") as f:
    skills_map = json.load(f)

def normalize_degrees(text):
    for key, value in degree_map.items():
        text = re.sub(fr"\b{re.escape(key)}\b", value, text, flags=re.IGNORECASE)
    return text

def normalize_skills(text):
    for key, value in skills_map.items():
        text = re.sub(fr"\b{re.escape(key)}\b", value, text, flags=re.IGNORECASE)
    return text

def basic_cleaning(text):
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special characters
    text = text.lower().strip()  # lowercase
    text = normalize_degrees(text)
    text = normalize_skills(text)
    return text

def tokenize_lemmatize(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_alpha and token.lemma_ not in stop_words:
            tokens.append(token.lemma_)
    return tokens

def preprocess_text(text):
    text = basic_cleaning(text)
    tokens = tokenize_lemmatize(text)
    return " ".join(tokens)
