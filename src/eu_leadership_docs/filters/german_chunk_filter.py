import de_core_news_sm
import logging
from pathlib import Path
import pandas as pd
import re
import spacy
from eu_leadership_docs.config import configure_logging
from eu_leadership_docs.utils.helpers import sent_tokenize, extract_context, clean_text, tokenize_lemmatize_text, filtered_path

# some configurations: logging, spacy, file paths, etc.
configure_logging()
logger = logging.getLogger(__name__)
nlp = spacy.load("de_core_news_sm")
nlp = de_core_news_sm.load()
file = Path(__file__).resolve().parents[3] / "data" / "raw" / "german_press_releases.csv"
ger_df = pd.read_csv(file)
df = ger_df.copy()
logger.info("Loaded German press releases!")
OUTPUT_CSV = filtered_path("german_press_releases_filtered.csv")

# Define the keyword list
keyword_list = [
    'russland', 'nawalny', 'russisch', 'moskau', 'putin', 'wladimir', 'kreml',
    'kiew', 'invasion', 'ukraine', 'sanktion', 'embargo', 'gas'
]

# Step 1: Clean the 'full_text' column
df['full_text'] = df['full_text'].apply(clean_text)

stems = ["russ", "mosk", "navalny","putin", "kreml", "ukrain", "embargo", "kiew", "donet", "sanktion", "gas", "rubel"]
pattern = re.compile("|".join(stems), re.IGNORECASE)

# Filter rows in 'full_text' based on the pattern
df['relevant_prs'] = df['full_text'].apply(lambda x: x if pd.notna(x) and pattern.search(x) else None)
logging.info("Filtered relevant press releases based on keywords and created 'relevant_prs' column.")
# Log the number of non-empty rows in 'relevant_prs'
non_empty_relevant_prs_count = df['relevant_prs'].notna().sum()
logging.info(f"Number of non-empty rows in 'relevant_prs': {non_empty_relevant_prs_count}")

# Step 3: Perform sentence splitting on 'relevant_prs'
df['text_sentences'] = df['relevant_prs'].apply(
    lambda x: sent_tokenize(x, spacy_model="de_core_news_sm") if pd.notna(x) else [])
logging.info("Successfully processed and added text_sentences column.")
# Step 4: Perform tokenization and lemmatization on 'text_sentences'
df['text_lemmatized'] = df['text_sentences'].apply(
    lambda sentences: [token for sentence in sentences for token in tokenize_lemmatize_text(sentence, spacy_model="de_core_news_sm")])
logging.info("Successfully processed and added text_lemmatized column.")
# Step 5: Extract relevant chunks from 'text_sentences' using the keyword list
df['context_sentences'] = df.apply(
    lambda row: extract_context(row, keyword_list=keyword_list) if pd.notna(row['relevant_prs']) else [],
    axis=1)
logging.info("Processing completed successfully.")
df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Saved {len(df)} records to {OUTPUT_CSV}")