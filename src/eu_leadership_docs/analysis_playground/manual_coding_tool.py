# """ A tool that would allow me to save the 300 press releases in an excel file, 
# so that I can save it and then do the manual coding on it.
# The coded dataset will not be needed in the analysis, 
# but the dictionary formed as a result of manual coding will be needed.
# So, the coded dataset will be uploaded, but only dictionary will be used in the further analysis.
# Coded dataset will serve as proof of coding and dictionary forming processes.

# What the tool should do:
# 1. access the datasets (this can be hard coded, since I will be using it once)
# 2. Extract the relevant columns (date, relevant chunks)
# 3. Group by year
# 4. take 10 random samples from each year per dataset (french, german, eeas)
# 5. add the 'actor' column to the dataset, which will indicate the source of the press release chunk (french, german, eeas)
# 6. save the dataset in an excel file, with the name "manual_coding_dataset.xlsx"
# """

# import pandas as pd
# from pathlib import Path
# from eu_leadership_docs.config import configure_logging
# import logging
# configure_logging()
# logger = logging.getLogger(__name__)

# def create_manual_coding_dataset():
#     fr_file = Path(__file__).resolve().parents[3] / "data" / "translated" / "french_translated_responses.csv"
#     fre_df = pd.read_csv(fr_file)
#     fr_df = fre_df.copy()

#     ger_file = Path(__file__).resolve().parents[3] / "data" / "translated" / "german_translated.csv"
#     gerr_df = pd.read_csv(ger_file)
#     ger_df = gerr_df.copy()

#     eeas_file = Path(__file__).resolve().parents[3] / "data" / "filtered" / "eeas_filtered_cleaned.csv"
#     eeass_df = pd.read_csv(eeas_file)
#     eeas_df = eeass_df.copy()
#     # now i need to process the datasets since i extracted the date in different ways...
#     # French dataset
#     # (DD/MM/YYYY)
#     fr_df['Date'] = pd.to_datetime(fr_df['Date'], format='%d/%m/%Y', errors="coerce")
#     fr_df = fr_df[fr_df['context_sentences'] != '[]']
#     fr_df = fr_df.sort_values(by='Date', ascending=True).reset_index(drop=True)
#     fr = fr_df[['Date', 'translated_context_sentences']].copy()
#     fr.rename(columns = {'translated_context_sentences':'context_sentences'}, inplace = True)    
#     fr['actor'] = 'france'
#     # German dataset
#     # (DD.MM.YYYY)
#     ger_df['Date'] = pd.to_datetime(ger_df['date'], format='%d.%m.%Y', errors="coerce")
#     ger_df = ger_df.sort_values(by='Date', ascending=True).reset_index(drop=True)
#     ger = ger_df[['Date', 'translated_context_sentences']].copy()
#     ger.rename(columns = {'translated_context_sentences':'context_sentences'}, inplace = True)
#     ger['actor'] = 'germany'
#     # EEAS dataset
#     # (DD.MM.YYYY)
#     eeas_df['Date'] = pd.to_datetime(eeas_df['article_date'], format='%d.%m.%Y', errors="coerce")
#     eeas_df = eeas_df.sort_values(by='Date', ascending=True).reset_index(drop=True)
#     eeas = eeas_df[['Date', 'context_sentences']].copy()
#     eeas['actor'] = 'eeas'
#     full_df = pd.concat([fr, ger, eeas], ignore_index=True)
#     full_df.to_csv(Path(__file__).resolve().parents[3] / "data" / "final" / "aligned_dataset.csv")


#     def sample_yearly(df, date_col, sample_size=2):
#         grouped = df.groupby(df[date_col].dt.year)
        
#         sampled = []
#         for year, group in grouped:
#             n_samples = min(len(group), sample_size)
#             sampled.append(group.sample(n=n_samples, random_state=33))
#         #first round (10 samples per year): 7
#         #second round (6 samples per year): 42
#         #unseen data 1 (1 sample per year): 9
#         #unseen data 2 (2 samples per year): 33
#         return pd.concat(sampled, ignore_index=True)

#     fr_sample = sample_yearly(fr, 'Date')
#     ger_sample = sample_yearly(ger, 'Date')
#     eeas_sample = sample_yearly(eeas, 'Date')

#     combined_df = pd.concat([fr_sample, ger_sample, eeas_sample], ignore_index=True)
#     combined_df.to_excel(Path(__file__).resolve().parents[3] / "data" / "second_unseen_data_dataset.xlsx", index=False)
#     logger.info("Manual coding dataset created and saved as 'second_unseen_data_dataset.xlsx'.")
#     return combined_df

# # call the function
# combined_df = create_manual_coding_dataset()
# #I just want to check how many samples we have from each actor per year
# print(combined_df.groupby([combined_df['Date'].dt.year, 'actor']).size())
