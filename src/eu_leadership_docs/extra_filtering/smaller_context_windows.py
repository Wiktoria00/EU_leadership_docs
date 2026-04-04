# here code connected to smaller context windows and russia being mentioned more than once, or not at all
import logging
from pathlib import Path
import pandas as pd
from eu_leadership_docs.config import configure_logging
from eu_leadership_docs.utils.helpers import final_path
import re
import ast
configure_logging()
logger = logging.getLogger(__name__)

# input and output files
INPUT_FILE = Path(__file__).resolve().parents[3] / "data" / "final" / "aligned_dataset.csv"
OUTPUT_CSV = final_path("final_dataset.csv")

# functions and keywords
def extract_context(row, keyword_list=list):
    keyword_set = {kw.lower() for kw in keyword_list}
    original_sentences = row['context_sentences']
    logging.debug(f"Extracting smaller context for row {row.name} with {len(original_sentences)} sentences.")
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
        for offset in range(-1, 3): # before was (-2, 5) so 2 before 4 after, now 1 before 2 after
            candidate = idx + offset
            if 0 <= candidate < len(original_sentences):
                context_indices.add(candidate)
    
    context_sentences = [original_sentences[i] for i in sorted(context_indices)]
    return context_sentences

keyword_list = ['russia','navalny','russian','moscow','putin','vladimir','kremlin',
    'kyiv','invasion','ukraine','sanction','embargo','gas']

# the script starts here
df = pd.read_csv(INPUT_FILE)
df = df.copy()
print(df.head())
# df['context_sentences'] = df['context_sentences'].apply(
#     lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# df['smaller_context_sentences'] = df.apply(extract_context, keyword_list=keyword_list, axis=1)


df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Saved {len(df)} records to {OUTPUT_CSV}")