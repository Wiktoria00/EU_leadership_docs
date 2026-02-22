import pandas as pd
from deep_translator import GoogleTranslator
import time
import re

#IMPORT SAVING LOGIC FROM PATH HELPERS (DATA/TRANSLATED)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Remove leading/trailing quotes
    text = text.strip().strip('"').strip("'")

    # 2. Replace newlines (\n, \r, etc.) with a space
    text = text.replace("\n", " ").replace("\r", " ")

    # 3. Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    # 4. (Optional) Remove weird non-breaking spaces or invisible chars
    text = text.replace("\xa0", " ").strip()

    return text

df = pd.read_csv("french_press_releases.csv")  ### CHANGE PATH HERE
df2 = df.copy()

def translate_long_text(text, max_chunk=4900):
    if not text or not isinstance(text, str):
        return ""

    translated_parts = []
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i+max_chunk]
        try:
            translated = GoogleTranslator(source='fr', target='en').translate(chunk)
            translated_parts.append(translated)
        except Exception as e:
            print(f"Error translating chunk: {e}")
            time.sleep(1)
            translated_parts.append("")
    return " ".join(translated_parts)


df2["clean_text"] = df2["Texte"].fillna("").apply(clean_text)
df2["translated"] = df2["clean_text"].apply(translate_long_text)

df2.to_csv("french_translated2.csv", index=False, encoding="utf-8")

