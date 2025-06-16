import spacy
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake


nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
rake = Rake()
