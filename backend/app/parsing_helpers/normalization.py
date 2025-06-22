# normalization.py
import json
import os

# Load Ontology Map
with open("config/skills_ontology.json", "r") as f:
    skill_ontology = json.load(f)

# Load Master Skill List (canonical skills)
with open("config/skill_list.json", "r") as f:
    master_skills = json.load(f)

# Reverse index for fast ontology lookup
reverse_skill_map = {}
for canonical, variants in skill_ontology.items():
    for variant in variants:
        reverse_skill_map[variant.lower()] = canonical

def ontology_normalize(skill):
    """First: Normalize based on strict ontology mapping"""
    skill = skill.lower().strip()
    return reverse_skill_map.get(skill, None)

def fuzzy_normalize(skill):
    """
    Second: Apply fuzzy substring matching using skill list if not mapped via ontology.
    """
    skill = skill.lower().strip()
    for canonical_skill in master_skills:
        if canonical_skill.lower() in skill:
            return canonical_skill
    return None  # If no match found

def normalize_skill(skill):
    """
    Hybrid Normalizer: Ontology first, then fuzzy keyword match, else fallback to raw skill.
    """
    normalized = ontology_normalize(skill)
    if normalized:
        return normalized
    normalized = fuzzy_normalize(skill)
    if normalized:
        return normalized
    return skill  # Return original if no mapping found

def normalize_skill_set(skill_list):
    """
    Normalize a list or set of skills at once.
    """
    return set([normalize_skill(skill) for skill in skill_list])
