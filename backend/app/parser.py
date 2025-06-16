import os
import pdfplumber
import docx
import io
import numpy as np
import nltk
from app.config.models import nlp, model, rake
from parsing_helpers.preprocessing import preprocess_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from parsing_helpers.gpt_extraction import extract_resume_info, extract_job_description_info
from parsing_helpers.spacy_extraction import spacy_extract_resume,skill_matcher
nltk.download('punkt')

model = SentenceTransformer('all-MiniLM-L6-v2') 

for resource in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)
# Load the model once

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
            return preprocess_text(text)
    except:
        return preprocess_text(extract_text_from_docx(file_bytes))

def parse_job_description(file_bytes):
    try:
        text = file_bytes.decode("utf-8")  # If it's a .txt
        if text.strip():
            return preprocess_text(text.strip())
    except UnicodeDecodeError:
        return preprocess_text(extract_text_from_docx(file_bytes))  # If it's a .docx

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


def semantic_skill_matcher(job_skills, resume_skills, threshold=0.75):
    matched_skills = set()

    for job_skill in job_skills:
        job_embedding = model.encode(job_skill)

        for resume_skill in resume_skills:
            resume_embedding = model.encode(resume_skill)
            score = cosine_similarity([job_embedding], [resume_embedding])[0][0]

            if score >= threshold:
                matched_skills.add(job_skill)
                break  # once matched, stop comparing this job skill

    missing_skills = job_skills - matched_skills
    return matched_skills, missing_skills


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
        resume_data = extract_resume_info(parsed_resume)
        job_data = extract_job_description_info(parsed_job)

        resume_skills = set(resume_data["skills"])
        job_skills = set(job_data["required_skills"])

        resume_experience = set(resume_data["experience"])
        job_experience = set(job_data["required_experience"])

        resume_education = set(resume_data["education"])
        job_education = set(job_data["required_education"])

        spacy_result = spacy_extract_resume(parsed_resume)
        spacy_skills = skill_matcher(parsed_job)

        final_resume_skills = resume_skills.union(spacy_skills)


        matched_skills, missing_skills = semantic_skill_matcher(job_skills, final_resume_skills)
        missing_experience = job_experience - resume_experience
        missing_education = job_education - resume_education


        return round(float(similarity) * 100, 2), missing_skills, missing_experience, missing_education, matched_skills
    except Exception as e:
        raise RuntimeError(f"Embedding match failed: {str(e)}")
