import json
import pdfplumber
import docx
import io
import numpy as np
import nltk
from config.models import nlp, model, rake
from parsing_helpers.preprocessing import preprocess_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from parsing_helpers.gpt_extraction import extract_resume_info, extract_job_description_info
from parsing_helpers.spacy_extraction import spacy_extract_resume,spacy_skill_finder
from parsing_helpers.normalization import normalize_skill_set

nltk.download('punkt')

for resource in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)


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
                matched_skills.add(resume_skill)
                break  # once matched, stop comparing this job skill

    missing_skills = job_skills - matched_skills
    return matched_skills, missing_skills


def get_embedding(text):
    if len(text) > 32000:
        text = text[:32000]
    return model.encode(text)

def extract_skill_names(skills):
    return set(
        skill["name"] if isinstance(skill, dict) else skill
        for skill in skills
        if isinstance(skill, (str, dict))
    )

def match_resume_to_job(parsed_resume, parsed_job):
    try:
        processed_parsed_resume=preprocess_text(parsed_resume)
        processed_parsed_job=preprocess_text(parsed_job)

        resume_embedding = get_embedding(processed_parsed_resume)
        job_embedding = get_embedding(processed_parsed_job)
        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

        # --- Keyword Extraction ---
        # Combine or choose extraction method

        resume_data = extract_resume_info(parsed_resume)
        job_data = extract_job_description_info(parsed_job)

       # resume_skills = extract_skill_names(resume_data["skills"])
        #job_skills = extract_skill_names(job_data["required_skills"])

        resume_experience = set(resume_data["experience"])
        job_experience = set(job_data["required_experience"])

        resume_education = set(resume_data["education"])
        job_education = set(job_data["required_education"])

        spacy_result = spacy_extract_resume(processed_parsed_resume)
        spacy_resume_skills = spacy_skill_finder(processed_parsed_resume)
        spacy_job_skills = spacy_skill_finder(processed_parsed_job)

        norm_spacy_resume_skills = normalize_skill_set(spacy_resume_skills)
        norm_spacy_job_skills = normalize_skill_set(spacy_job_skills)

        #final_resume_skills = resume_skills.union(norm_spacy_resume_skills)
        #final_job_skills = job_skills.union(norm_spacy_job_skills)


        matched_skills, missing_skills = semantic_skill_matcher(norm_spacy_job_skills, norm_spacy_resume_skills)
        missing_experience = job_experience - resume_experience
        missing_education = job_education - resume_education
        print("\n\n🟡 spacy_resume_skills:\n", spacy_resume_skills) 
        print("\n\n🟡 spacy_job_skills:\n", spacy_job_skills) 
        print("\n\n🟡 norm_spacy_resume_skills:\n", norm_spacy_resume_skills) 
        print("\n\n🟡 norm_spacy_job_skills:\n", norm_spacy_job_skills) 

       
        return round(float(similarity) * 100, 2), missing_skills, missing_experience, missing_education, matched_skills, resume_data, job_data
    except Exception as e:
        raise RuntimeError(f"Embedding match failed: {str(e)}")

def format_match_results(matched, title="Matched Skills"):
    lines = [f"🟢 {title}"]
    for group in matched:
        cat = group.get("category", "General")
        items = group.get("items", [])
        reason = group.get("justification", "")
        lines.append(f"\n{cat}")
        for item in items:
            lines.append(f"• {item}")
        if reason:
            lines.append(f"\n{reason}")
    return "\n".join(lines)

def format_missing_results(missing, title="Missing Skills"):
    lines = [f"\n{title}"]
    for group in missing:
        cat = group.get("category", "General")
        items = group.get("items", [])
        description = group.get("description", "")
        lines.append(f"\n{cat}")
        for item in items:
            lines.append(f"• {item}")
        if description:
            lines.append(f"\n{description}")
    return "\n".join(lines)

def format_missing_experience(experience_set):
    if not experience_set:
        return ""
    lines = ["\nMissing Experience\n"]
    for exp in sorted(experience_set):
        lines.append(f"• {exp}")
    return "\n".join(lines)

def format_missing_education(education_set):
    if not education_set:
        return ""
    lines = ["\nMissing Education\n"]
    for edu in sorted(education_set):
        lines.append(f"• {edu}")
    return "\n".join(lines)
