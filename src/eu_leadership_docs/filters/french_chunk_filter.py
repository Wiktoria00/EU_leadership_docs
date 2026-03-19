import re
import pandas as pd
import logging
from pathlib import Path
import spacy
from fr_core_news_sm import load as load_fr

# Import your local config and helpers
from eu_leadership_docs.config import configure_logging
from eu_leadership_docs.utils.helpers import sent_tokenize, extract_context, clean_text, tokenize_lemmatize_text, filtered_path

# --- Configuration ---
configure_logging()
logger = logging.getLogger(__name__)

# Load French model
nlp = load_fr()

# File paths
INPUT_FILE = Path(__file__).resolve().parents[3] / "data" / "raw" / "french_press_releases.csv"
OUTPUT_CSV = filtered_path("french_press_releases_filtered_responses.csv")

# Keywords for filtering
keyword_list = [
    'russie', 'navalny', 'russe', 'moscou', 'poutine', 'vladimir', 'kremlin',
    'kiev', 'invasion', 'ukraine', 'sanction', 'embargo', 'gaz'
]

# Extract Answers from Q&A
def extract_qa_answers(text):
    """
    Extracts only the answers (R − ...) from French Q&A press releases.
    If no Q&A pattern is found, returns the original text.
    """
    if not pd.notna(text):
        return text
    
    # Normalize potential encoding issues in dashes (e.g., non-breaking spaces, different hyphens)
    # Pattern looks for 'R' followed by optional space, various dash types, optional space
    # Then captures everything until the next 'Q −' or end of string
    # Flags: re.DOTALL makes '.' match newlines, re.IGNORECASE for safety
    
    # Split by questions first to isolate blocks
    # We look for 'Q −' or 'Q -' as delimiters
    parts = re.split(r'\n?\s*Q\s*[−:-]\s*', text, flags=re.IGNORECASE)    
    processed_parts = []
    
    for part in parts:
        # In each part, look for the Answer marker 'R −'
        # We expect the structure inside a 'Q-split' to be potentially " [Question Text] R − [Answer Text] "
        # Or if the text starts directly with R (if the first Q was at the very beginning)
        
        # Regex to find 'R −' and capture everything after it
        match = re.search(r'\n?\s*R\s*[−:-]\s*(.*)', part, flags=re.DOTALL | re.IGNORECASE)
        
        if match:
            answer_content = match.group(1).strip()
            if answer_content:
                processed_parts.append(answer_content)
        else:
            # If there is text before the first 'Q' that doesn't have an 'R', 
            # it might be an intro paragraph. 
            # However, in strict Q&A transcripts, we usually only want the R parts.
            # If the part doesn't contain an 'R', we discard the question text.
            pass

    if processed_parts:
        return " ".join(processed_parts)
    
    # Fallback: If no Q&A pattern found, return original text (handles standard statements)
    return text

logger.info("Loading French press releases...")
df = pd.read_csv(INPUT_FILE)
logger.info(f"Loaded {len(df)} records.")

df = df.copy()

# 1. Clean Text (remove extra whitespace, fix encoding)
logger.info("Cleaning text...")
df['Texte'] = df['Texte'].apply(clean_text)

# 2. Extract Only Answers (Q&A Filtering)
logger.info("Extracting Q&A answers (filtering out questions)...")
df['Texte'] = df['Texte'].apply(extract_qa_answers)

# 3. Sentence Tokenization
logger.info("Tokenizing sentences...")
df['text_sentences'] = df['Texte'].apply(
    lambda x: sent_tokenize(x, spacy_model="fr_core_news_sm") if pd.notna(x) else []
)

# 4. Lemmatization (Tokenizing the whole chunk for keyword matching)
# Note: You are currently lemmatizing the whole list of sentences into one big list per row.
# This is fine for keyword filtering, but ensure your 'extract_context' handles lists of tokens correctly.
logger.info("Lemmatizing text...")
df['text_lemmatized'] = df['text_sentences'].apply(
    lambda sentences: [
        token 
        for sentence in sentences 
        for token in tokenize_lemmatize_text(sentence, spacy_model="fr_core_news_sm")
    ]
)

# 5. Extract Relevant Context (Keyword Filtering)
logger.info("Extracting relevant context chunks based on keywords...")
df['context_sentences'] = df.apply(
    lambda row: extract_context(row, keyword_list=keyword_list) if pd.notna(row['Texte']) else [],
    axis=1
)

# --- Save ---
df.to_csv(OUTPUT_CSV, index=False)
logger.info(f"Successfully saved {len(df)} records to {OUTPUT_CSV}")
logger.info("Processing complete.")