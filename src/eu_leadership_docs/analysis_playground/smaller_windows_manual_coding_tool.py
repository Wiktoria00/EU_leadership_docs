""" A tool that would allow me to save the 300 press releases in an excel file, 
so that I can save it and then do the manual coding on it.
The coded dataset will not be needed in the analysis, 
but the dictionary formed as a result of manual coding will be needed.
So, the coded dataset will be uploaded, but only dictionary will be used in the further analysis.
Coded dataset will serve as proof of coding and dictionary forming processes.

What the tool should do:
1. access the datasets (this can be hard coded, since I will be using it once)
2. Extract the relevant columns (date, relevant chunks)
3. Group by year
4. take 10 random samples from each year per dataset (french, german, eeas)
5. add the 'actor' column to the dataset, which will indicate the source of the press release chunk (french, german, eeas)
6. save the dataset in an excel file, with the name "manual_coding_dataset.xlsx"
"""

import pandas as pd
from pathlib import Path
from eu_leadership_docs.config import configure_logging
import logging
configure_logging()
logger = logging.getLogger(__name__)

def create_manual_coding_dataset():
    # 1. Access the dataset
    file = Path(__file__).resolve().parents[3] / "data" / "final" / "aligned_dataset.csv"
    df = pd.read_csv(file)
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    def sample_yearly_per_actor(df, date_col, actor_col, sample_size=16):
        # Group by BOTH year and actor
        grouped = df.groupby([df[date_col].dt.year, df[actor_col]])
        
        sampled = []
        for (year, actor), group in grouped:
            n_samples = min(len(group), sample_size)
            if n_samples > 0:
                sampled.append(group.sample(n=n_samples, random_state=7))
        if not sampled:
            return pd.DataFrame(columns=df.columns)
            
        return pd.concat(sampled, ignore_index=True)
    
    combined_df = sample_yearly_per_actor(df, 'Date', 'actor', sample_size=15)
    output_path = Path(__file__).resolve().parents[3] / "data" / "small_window_manual_coding.xlsx"
    combined_df.to_excel(output_path, index=False)
    logger.info(f"Manual coding dataset created with {len(combined_df)} rows.")
    return combined_df

combined_df = create_manual_coding_dataset()
#I just want to check how many samples we have from each actor per year
print(combined_df.groupby([combined_df['Date'].dt.year, 'actor']).size())
