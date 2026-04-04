import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from eu_leadership_docs.utils.helpers import translated_path
from pathlib import Path
import ast
import logging
from eu_leadership_docs.config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

# Load Helsinki-NLP French-to-English translation model
model_name = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

output_csv = translated_path("french_translated_responses.csv")
file = Path(__file__).resolve().parents[3] / "data" / "filtered" / "french_press_releases_filtered_responses.csv"
fr_df = pd.read_csv(file)
df = fr_df.copy()

def safe_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        logger.warning(f"Skipping malformed context_sentences: {value}")
        return []

def translate_sentences(sentences):
    # Check if sentences list is not empty
    if not sentences:
        logger.warning("Empty sentence list, skipping translation.")
        return []

    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        num_beams=5,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

df['context_sentences'] = df['context_sentences'].apply(safe_eval)
df['translated_context_sentences'] = df['context_sentences'].apply(translate_sentences)
df.to_csv(output_csv, index=False)
logging.info(f"Saved {len(df)} records to {output_csv}")
