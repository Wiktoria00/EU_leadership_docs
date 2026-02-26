## COLUMN NAMES !! 

import fr_core_news_sm
import logging
from pathlib import Path
import pandas as pd
import spacy
from eu_leadership_docs.config import configure_logging
from eu_leadership_docs.utils.helpers import sent_tokenize, extract_context, clean_text, tokenize_lemmatize_text, filtered_path

# some configurations: logging, spacy, file paths, etc.
configure_logging()
logger = logging.getLogger(__name__)
nlp = spacy.load("fr_core_news_sm")
nlp = fr_core_news_sm.load()
file = Path(__file__).resolve().parents[3] / "data" / "raw" / "french_press_releases.csv"
ger_df = pd.read_csv(file)
df = ger_df.copy()
logger.info("Loaded French press releases!")
OUTPUT_CSV = filtered_path("french_press_releases_filtered.csv")

keyword_list = ['russie','navalny','russe','moscou','poutine','vladimir','kremlin',
    'kiev','invasion','ukraine','sanction','embargo','gaz']

df['Texte'] = df['Texte'].apply(clean_text)
df['text_sentences'] = df['Texte'].apply(
    lambda x: sent_tokenize(x, spacy_model="fr_core_news_sm") if pd.notna(x) else [])
logging.info("Successfully processed and added text_sentences column.")

'''
Time for chunks! This includes:
1) tokenising and lemmatization, 
2) filtering for relevant sentences, 
3) then putting relevant chunks to a new column (otherwise nan)
'''
df['text_lemmatized'] = df['text_sentences'].apply(
    lambda sentences: [token for sentence in sentences for token in
                        tokenize_lemmatize_text(sentence, spacy_model="fr_core_news_sm")])
logging.info("Successfully processed and added text_lemmatized column.")

df['context_sentences'] = df.apply(
    lambda row: extract_context(row, keyword_list=keyword_list) if pd.notna(row['Texte']) else [],
    axis=1)
logging.info("Processing completed successfully.")

df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Saved {len(df)} records to {OUTPUT_CSV}")
