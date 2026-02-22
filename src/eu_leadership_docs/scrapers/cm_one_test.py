import pdfplumber
from pathlib import Path

"""The following code is a test to see if I can read the pdfs and extract text from them,
 in a similar manner to the french scraper. Feel free to run the code.
As one can see, unfortunately, the pdfs are not structured in a way that allows us to extract text,
 so I will need to analyse them manually."""

file = Path(__file__).resolve().parents[3] / "data" / "pdfs" / "M__ST-9-2014-INIT_en.pdf"
with pdfplumber.open(file) as pdf:
    pages = [p.extract_text() or '' for p in pdf.pages]
    print(f"Number of pages: {len(pages)}")
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        print(f"page {i + 1} text:\n{text[:500] if text else 'No text found'}\n{'-' * 40}")
        print(f"page {i + 1} objects: {page.objects}\n{'-' * 40}")

pdf_text = "\n".join(pages)
print("Full concat text:\n", pdf_text[:1000],"...")
