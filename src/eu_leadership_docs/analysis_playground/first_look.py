import pandas as pd
from pathlib import Path
import logging
from eu_leadership_docs.config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

fr_file = Path(__file__).resolve().parents[3] / "data" / "translated" / "french_translated.csv"
fre_df = pd.read_csv(fr_file)
fr_df = fre_df.copy()

ger_file = Path(__file__).resolve().parents[3] / "data" / "translated" / "german_translated.csv"
gerr_df = pd.read_csv(ger_file)
ger_df = gerr_df.copy()

eeas_file = Path(__file__).resolve().parents[3] / "data" / "filtered" / "eeas_filtered_cleaned.csv"
eeass_df = pd.read_csv(eeas_file)
eeas_df = eeass_df.copy()

minutes = Path(__file__).resolve().parents[3] / "data" / "raw" / "council_minutes_data.xlsx"
minutes_df = pd.read_excel(minutes)
m_df = minutes_df.copy()

m_df['meeting_date'] = pd.to_datetime(m_df['meeting_date'])
m_df.sort_values(by='meeting_date')

#change to french df
fr_df['Date'] = pd.to_datetime(fr_df['Date'], errors="coerce")
fr_df = fr_df.sort_values(by='Date', ascending=True).reset_index(drop=True)

#change to eeas df
eeas_df["article_date"] = pd.to_datetime(eeas_df["article_date"], errors="coerce")
eeas_df = eeas_df.sort_values("article_date", ascending=True).reset_index(drop=True)

for index in range(2):
    print(f"FRENCH PRESS RELEASE DATE: {fr_df['Date'][index]}.Relevant sentences: {fr_df['translated_context_sentences'][index]}")
    print(f"GERMAN PRESS RELEASE DATE: {ger_df['date'][index]}. Relevant sentences: {ger_df['translated_context_sentences'][index]}")
    print(f"EEAS PRESS RELEASE DATE: {eeas_df['article_date'][index]}. relevant sentences: {eeas_df['context_sentences'][index]}")
    print(f"COUNCIL MINUTES DATE: {m_df['meeting_date'][index]}. Relevant text: {m_df['text'][index]}")