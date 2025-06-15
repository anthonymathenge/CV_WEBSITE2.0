import os
import pdfplumber
import docx
import io
import numpy as np
import spacy
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
nltk.download('punkt')

for resource in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)
# Load the model once
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
rake = Rake()

def extract_text_from_pdf(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = " ".join([page.extract_text() or "" for page in pdf.pages])
        return text.strip()

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def parse_resume(file_bytes):
    try:
        text = extract_text_from_pdf(file_bytes)
        if text:
            return text
    except:
        return extract_text_from_docx(file_bytes)

def parse_job_description(file_bytes):
    try:
        text = file_bytes.decode("utf-8")  # If it's a .txt
        if text.strip():
            return text.strip()
    except UnicodeDecodeError:
        return extract_text_from_docx(file_bytes)  # If it's a .docx

def extract_keywords_spacy(text: str) -> set:
    """
    Extract keywords using spaCy noun chunks and named entities,
    filtered and normalized for accuracy.
    """
    doc = nlp(text)
    keywords = set()

    # Extract noun chunks (multi-word phrases)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        # Filter out short/noisy chunks
        if len(phrase) > 2 and not any(tok.is_stop or tok.is_punct for tok in chunk):
            keywords.add(phrase)

    # Extract named entities (PERSON, ORG, GPE, PRODUCT, etc.)
    for ent in doc.ents:
        ent_text = ent.text.lower().strip()
        if len(ent_text) > 2:
            keywords.add(ent_text)

    return keywords

def extract_keywords_rake(text: str, max_keywords=20) -> list:
    """
    Extract keywords/phrases using RAKE algorithm as a supplement.
    Returns a list of keywords sorted by score.
    """
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()[:max_keywords]
    return [phrase.lower() for phrase in ranked_phrases]


def get_embedding(text):
    if len(text) > 32000:
        text = text[:32000]
    return model.encode(text)

def match_resume_to_job(parsed_resume, parsed_job):
    try:
        resume_embedding = get_embedding(parsed_resume)
        job_embedding = get_embedding(parsed_job)
        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

        # --- Keyword Extraction ---
        # Combine or choose extraction method
        resume_keywords = extract_keywords_spacy(parsed_resume).union(
            extract_keywords_rake(parsed_resume)
        )
        job_keywords = extract_keywords_spacy(parsed_job).union(
            extract_keywords_rake(parsed_job)
        )

        missing_keywords = sorted(job_keywords - resume_keywords)

        return round(float(similarity) * 100, 2), missing_keywords
    except Exception as e:
        raise RuntimeError(f"Embedding match failed: {str(e)}")
