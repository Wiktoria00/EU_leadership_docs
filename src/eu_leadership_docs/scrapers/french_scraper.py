import pandas as pd
import re
import pdfplumber

all_text = []

pdf_location = "scrapers/mon_fichier_pdf_2014-2025.pdf"

with pdfplumber.open(pdf_location) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)

# Combine all pages into a single string if needed
pdf_text = "\n".join(all_text)

with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(pdf_text)

with open("pdf_text.txt", "r", encoding="utf-8") as f:
    pdf_text = f.read()

# Split entries by the "Point de presse" keyword
entries = re.split(r'\d+ / \d+ − Point de presse du ', pdf_text)[1:]  # skip header

data = []
for entry in entries:
    # Extract date
    date_match = re.match(r'(\d{2}/\d{2}/\d{4})', entry)
    date = date_match.group(1) if date_match else None
    
    # Extract reference
    ref_match = re.search(r'Référence (\S+)', entry)
    reference = ref_match.group(1) if ref_match else None
    
    # Extract texte
    texte_match = re.search(r'Texte (.+)', entry, re.DOTALL)
    texte = texte_match.group(1).strip() if texte_match else None
    
    # Type
    type_ = "Point de presse"
    
    data.append({
        "Date": date,
        "Reference": reference,
        "Texte": texte,
        "Type": type_
    })

# Create DataFrame
df = pd.DataFrame(data)
df.head()
#df.to_csv('french_press_releases2.csv', index= False)
