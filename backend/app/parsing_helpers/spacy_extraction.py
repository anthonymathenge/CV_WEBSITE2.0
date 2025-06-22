import re
import spacy
import json 
nlp = spacy.load("en_core_web_trf")
from spacy.matcher import PhraseMatcher

with open("config/skill_list.json", "r") as f:
    skills_list = json.load(f)# Expand as needed


def spacy_extract_resume(text):
    doc = nlp(text)

    extracted = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": set(),
        "education": set(),
        "experience": set()
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not extracted["name"]:
            extracted["name"] = ent.text
        elif ent.label_ == "ORG":
            extracted["education"].add(ent.text)
        elif ent.label_ == "DATE":
            extracted["experience"].add(ent.text)
        elif ent.label_ == "GPE":
            extracted["education"].add(ent.text)  # sometimes locations overlap with schools

    return extracted


def spacy_skill_finder(text):
    skills = []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        if re.search(pattern, text, re.IGNORECASE):
            skills.append(skill)
    return skills