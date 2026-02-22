import de_core_news_sm
import logging
from pathlib import Path
import pandas as pd
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


# we clean our full text column and create a new column with sentences
try:
    df['full_text'] = df['full_text'].apply(clean_text)
    df['text_sentences'] = df['full_text'].apply(sent_tokenize, spacy_model= "de_core_news_sm")
    logging.info("Successfully processed and added full_text_sentences column.")
except Exception as e:
    logging.error(f"An error occurred: {e}")

'''
Time for chunks! This includes:
1) tokenising and lemmatization, 
2) filtering for relevant sentences, 
3) then putting relevant chunks to a new column (otherwise nan)
'''

## chunks step 1, so tokens and lemmatization
# Process the text_sentences column to create the text_lemmatized column
try:
    df['text_lemmatized'] = df['text_sentences'].apply(
        lambda sentences: [token for sentence in sentences for token in tokenize_lemmatize_text(sentence, spacy_model="de_core_news_sm")]
    )
    logging.info("success! processed and added text_lemmatized column.")
except Exception as e:
    logging.error(f"ERROR:( . It occurred while lemmatizing text: {e}")
## chunks step 2, so filtering for sentences with keywords and creating a new column 
    # (plus padding, so 2 sentences before and 4 senctences after the relevant one)
    # we will be creating the column as we go
keyword_list = [
    'russland', 'nawalny', 'russisch', 'moskau', 'putin', 'wladimir', 'kreml',
    'kiew', 'invasion', 'ukraine', 'sanktion', 'embargo', 'gas']
df['context_sentences'] = df.apply(extract_context, keyword_list=keyword_list, axis=1)

df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Saved {len(df)} records to {OUTPUT_CSV}")
#logging.info("German chunks filter script finished, wohoo!")