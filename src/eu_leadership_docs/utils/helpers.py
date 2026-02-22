from pathlib import Path
from eu_leadership_docs import config
import re
import pandas as pd
import spacy
import logging
from eu_leadership_docs.config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

# path helpers
def get_data_path(subdir: str, filename: str) -> Path:
    """Get path to file in data subdir."""
    if subdir == "raw":
        return config.RAW_DIR / filename
    elif subdir == "translated":
        return config.TRANSLATED_DIR / filename
    elif subdir == "filtered":
        return config.FILTERED_DIR / filename
    else:
        raise ValueError(f"Unknown subdir: {subdir}")

def raw_path(filename: str) -> Path:
    return get_data_path("raw", filename)

def translated_path(filename: str) -> Path:
    return get_data_path("translated", filename)

def filtered_path(filename: str) -> Path:
    return get_data_path("filtered", filename)

# text preprocessing helpers

def sent_tokenize(text, spacy_model= str):
    if pd.isna(text):
        return ""
    doc = spacy.load(spacy_model)(text)
    return [sent.text.strip() for sent in doc.sents]


def clean_text(text):
    if not isinstance(text, str):
        return ""
    # leading/trailing quotes
    text = text.strip().strip('"').strip("'")
    # newlines
    text = text.replace("\n", " ").replace("\r", " ")
    # multiple spaces
    text = re.sub(r"\s+", " ", text)
    # weird non-breaking spaces or invisible chars
    text = text.replace("\xa0", " ").strip()
    return text

def tokenize_lemmatize_text(text, spacy_model=str):
    if pd.isna(text):
        return []
    nlp = spacy.load(spacy_model)
    lemmatized_tokens = []
    for sentence in sent_tokenize(text, spacy_model=spacy_model):
        doc = nlp(sentence)
        lemmatized_tokens.extend([token.lemma_ for token in doc if not token.is_punct and not token.is_space])
    return lemmatized_tokens

def extract_context(row, keyword_list=list):
    keyword_set = {kw.lower() for kw in keyword_list}
    original_sentences = row['text_sentences']
    logging.debug(f"Extracting context for row {row.name} with {len(original_sentences)} sentences.")
    if not original_sentences:
        return []
    
    matching_indices = set()
    for i, sentence in enumerate(original_sentences):
        sentence_lower = sentence.lower()
        for kw in keyword_set:
            # Match whole word only (word boundaries)
            if re.search(r'\b' + re.escape(kw) + r'\b', sentence_lower):
                matching_indices.add(i)
                break  # no need to check other keywords for this sentence
    
    if not matching_indices:
        return []
    
    context_indices = set()
    for idx in matching_indices:
        for offset in range(-2, 5):
            candidate = idx + offset
            if 0 <= candidate < len(original_sentences):
                context_indices.add(candidate)
    
    context_sentences = [original_sentences[i] for i in sorted(context_indices)]
    return context_sentences