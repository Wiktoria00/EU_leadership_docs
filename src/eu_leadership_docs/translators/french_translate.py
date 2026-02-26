
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from eu_leadership_docs.utils.helpers import translated_path
from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")

# start of test use of the tranlation model
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
# end of test use of the translation model

output_csv = translated_path("french_translated.csv")
file = Path(__file__).resolve().parents[3] / "data" / "raw" / "french_filtered_cleaned.csv"
fr_df = pd.read_csv(file)
df = fr_df.copy()


# CODE BELOW NEEDS TO BE FIXED TO CORRESPOND TO QWEN MODEL , NOT GOOGLE TRANSLATE.

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

