#imports
import pandas as pd
import logging
import numpy as np
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import ast
import spacy
from eu_leadership_docs.config import configure_logging
nlp = spacy.load('en_core_web_sm')
configure_logging()
logger = logging.getLogger(__name__)

# File paths
MANUAL_DATASET_FILE = Path(__file__).resolve().parents[3] / "data" / "manual_coding_dataset.xlsx"

manual_dataset_original = pd.read_excel(MANUAL_DATASET_FILE)
manual_dataset = manual_dataset_original.copy()

keyword_lists = {
    'realist': ["sanction", "violation", "annexation", "firmness", "illegal",
            "weapon", "deterrence", "compliance", "non-compliance", 
            "hybrid", "interference", "coercion",
            "unlawfully detain", "illegitimate", "troop", "punishment", 
            "joint investigation team", "resolution 2166", "accountability",
            "foreign agent", "inf treaty",
            "firm and strong", "firm response", "strong response", 
            "firm stance", "strong stance", "firm position", "mass detention",
            "dependence", "gas", "energy", "strong signal", "nord stream 2", "competitive", 
            "european army", "accountable", "restrictive measure", "restrictive", 
            "embargo", "tariffs", "disinformation", "separatist", "impose", 
            "destructive", "tough", "cyber attack", "repowereu", "military assistance", 
            "eumam", "moldova", "oil", "civil protection mechanism", "military support", 
            "provocation", "shadow fleet", "military escalation", "ceasefire violation", 
            "sovereignty","full implementation", "no longer desired","peace order",
            "demand dialogue", "judicial harassment", "freedom of the press",
            "rules-based order", "police brutality","unjustified war", "strong resolve",
            "black sea", "food insecurity", "war of aggression","russian aggression",
            "military resistance", "summon", "hold responsible", "political prisoner",
            "fully responsible", "penal colony"],
    'diplomatic': ["dialogue", "minsk", "diplomatic", "peaceful", "de-escalation", "consultation", "partner", 
                   "cessation", "humanitarian", "ease", "negotiate", "discussion", "meeting", "normandy", 
                   "ceasefire", "cooperation", "solution", "common ground", "relationship", "agree", "engage", 
                   "trilateral", "cooperation", "friend", "mediation", "counterpart", "discuss", "peace plan", 
                   "delegation", "trilateral contact group", "mutual", "alliance", "common interest", "euren", 
                   "constructive", "solidarity", "round table"],
    'EU_collective':["european family", "we european","european level", "at eu level", "european project","eumam", "neighbourhood policy", "eastern partnership", 
                     "energy union", "eastern partner", "eu engagement", "european crisis", "european platform", "association agreement", "accession",
                     "council of europe", "28 Member State", "27 Member State", "european partner", "european council", 
                     "european unification", "european interest", "collective decision by the european union", "by the european union", "enlargement",
                     "partners in the european union", "european decision-making", "european decision", "schengen","european counterpart",
                     "european solidarity", "europe 's security", "european peace","european political cooperation", "european engagement", "european restrictive", "european sanction"],
    'intensity':["must", "territorial integrity", "commit to", "mobilize", "firmness", "require to", 
                 "call on", "concern", "expect", "pressure", "necessary", "aggression","not have time",
                 "stress the","underline","crucial","immediately","illegal","terrible",
                 "urge","demand","unjustifiable","unacceptable","reiterate","serious","enforce","fail",
                 "essential","great dismay","great concern","regret","disaster","setback","stress","worry",
                 "cynical","attack","disregard","condemn","annexation", "reaffirm", "appeal","brutal","strongly condemn",
                 "extremely concern", "extremely worry","illegal annexation", "all mean", "human right violation", "human right",
                 "gravely concern","grave concern","cruel act","extremely worry","non-recognition policy",
                 "murderous persistence","deplore","denounce","outrage","strongest condemnation",
                 "underline concern","atrocity","strongest possible term",
                 "war crime","immense suffer","full-scale invasion","flagrant violation",
                 "unprovoke","ruthless","outrageous","heinous","blatant disregard"]
}

def extract_features(segment_list, keyword_lists):
    segment_text = " ".join(segment_list).lower()
    doc = nlp(segment_text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    lemmatized_text = ' '.join(lemmatized_tokens)
    total_words = len(lemmatized_tokens)
    if total_words == 0:
        return {f"{category}_rate": 0 for category in keyword_lists.keys()}
    features = {}
    for category, keywords in keyword_lists.items():
        keyword_count = 0
        for keyword in keywords:
            keyword_doc = nlp(keyword.lower())
            lemmatized_keyword = ' '.join([token.lemma_ for token in keyword_doc])
            pattern = r'\b' + re.escape(lemmatized_keyword) + r'\b'
            matches = re.findall(pattern, lemmatized_text)
            keyword_count += len(matches)
        rate = keyword_count / total_words
        features[f'{category}_rate'] = rate
    return features

def clean_html_tags(text):
    cleaned = re.sub(r'<[^>]+>', '', text)
    return cleaned

def derive_intensity_from_rates(intensity_rate):
    return 1 if intensity_rate > 0.001 else 0

# Filter noise
non_noise_mask = manual_dataset['issue distibution'] != 'noise'
manual_dataset_clean = manual_dataset[non_noise_mask].reset_index(drop=True)

#print(manual_dataset.columns.tolist())
#print(manual_dataset_clean['actor'].value_counts())

# Rebuild sample segments and manual codes
sample_segments = manual_dataset_clean['translated_context_sentences']
manual_codes = manual_dataset_clean[['intensity', 'eu_collective_framing', 
                                      'realist_stance', 'diplomatic_stance']].copy()
manual_codes['intensity'] = manual_codes['intensity'].apply(lambda x: 1 if x > 0 else 0)

# Create France/Germany only mask for EU collective framing
non_eeas_mask = manual_dataset_clean['actor'].str.lower() != 'eeas'
X_df_non_eeas_idx = manual_dataset_clean[non_eeas_mask].index  # keep original indices
sample_segments_non_eeas = sample_segments[non_eeas_mask].reset_index(drop=True)
manual_codes_non_eeas = manual_codes[non_eeas_mask].reset_index(drop=True)

#print(f"\nFrance/Germany segments for EU collective training: {non_eeas_mask.sum()}")
#print(manual_codes_non_eeas['eu_collective_framing'].value_counts())

# Build feature matrix for ALL segments
feature_matrix = []
for segment_string in sample_segments:
    segment_list = ast.literal_eval(segment_string)
    segment_list_cleaned = [clean_html_tags(sentence) for sentence in segment_list]
    features = extract_features(segment_list_cleaned, keyword_lists)
    features_row = [
        features['realist_rate'],
        features['diplomatic_rate'],
        features['EU_collective_rate'],
        features['intensity_rate']
    ]
    feature_matrix.append(features_row)

X = np.array(feature_matrix)
print(X.shape)

X_df = pd.DataFrame(X)
X_df.columns = ['realist_rate', 'diplomatic_rate', 'EU_collective_rate', 'intensity_rate']

# Build feature matrix for France/Germany only (for EU collective)
X_df_non_eeas = X_df[non_eeas_mask].reset_index(drop=True)

feature_cols = {
    'intensity': ['intensity_rate'],
    'eu_collective_framing': ['EU_collective_rate'],
    'realist_stance': ['realist_rate'],
    'diplomatic_stance': ['diplomatic_rate']
}

derived_intensity = X_df.apply(
    lambda row: derive_intensity_from_rates(row['intensity_rate']),
    axis=1
)

classifiers = {}
results = {}
param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'solver': ['lbfgs'],
    'max_iter': [1000, 2000]
}

for dimension in ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']:
    logger.info(f"Training the classifier for {dimension}")

    cols = feature_cols[dimension]

    # EU collective framing: train on France/Germany only
    if dimension == 'eu_collective_framing':
        X_dim = X_df_non_eeas[cols].values
        y = np.array(manual_codes_non_eeas[dimension])
    else:
        X_dim = X_df[cols].values
        y = np.array(manual_codes[dimension])

    X_train, X_test, y_train, y_test = train_test_split(
        X_dim, y, test_size=0.2, random_state=7, stratify=y
    )

    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=7, class_weight='balanced'),
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    logger.info(f"Best parameters for {dimension}: {best_params}")

    clf = grid_search.best_estimator_
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    logger.info(f"Performance for {dimension} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    classifiers[dimension] = clf
    results[dimension] = {
        'best_params': best_params,
        'best_cv_score': grid_search.best_score_,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'feature_cols': cols
    }

# Validation on full coded sample
logger.info("Validating classifiers on the full coded sample dataset...")
validation_results = {}
for dimension in ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']:
    clf = classifiers[dimension]
    cols = feature_cols[dimension]

    # EU collective framing: validate on France/Germany only
    if dimension == 'eu_collective_framing':
        X_dim = X_df_non_eeas[cols].values
        y = np.array(manual_codes_non_eeas[dimension])
    else:
        X_dim = X_df[cols].values
        y = np.array(manual_codes[dimension])

    y_pred = clf.predict(X_dim)
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='weighted', zero_division=0
    )
    validation_results[dimension] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

print("\nValidation results on the full coded sample dataset:")
for dimension, metrics in validation_results.items():
    print(f"{dimension.capitalize()}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1_score']:.4f}")

# Rule-based intensity comparison
y_true = np.array(manual_codes['intensity'])
y_rule = np.array([
    derive_intensity_from_rates(row['intensity_rate'])
    for _, row in X_df.iterrows()
])

acc = accuracy_score(y_true, y_rule)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_rule, average='weighted', zero_division=0)
print(f"\nRule-based intensity: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


# showing final output: 
# Add this at the end, after the rule-based intensity comparison

# EU collective framing detailed validation
eu_present = manual_codes_non_eeas['eu_collective_framing'] == 1
eu_zero_rate = X_df_non_eeas['EU_collective_rate'] == 0

print("\nEU Collective Framing Coverage (France/Germany only):")
print(f"  Coded present AND rate > 0: {(eu_present & ~eu_zero_rate).sum()}")
print(f"  Coded present BUT rate = 0: {(eu_present & eu_zero_rate).sum()}")
print(f"  Coded absent AND rate > 0: {(~eu_present & ~eu_zero_rate).sum()}")
print(f"  Coded absent AND rate = 0: {(~eu_present & eu_zero_rate).sum()}")
print(f"\nLabel distribution: {dict(manual_codes_non_eeas['eu_collective_framing'].value_counts())}")

clf = classifiers['eu_collective_framing']
X_dim = X_df_non_eeas[feature_cols['eu_collective_framing']].values
y_true_eu = np.array(manual_codes_non_eeas['eu_collective_framing'])
y_pred_eu = clf.predict(X_dim)

print("\n=== EU COLLECTIVE FRAMING (France/Germany only) ===")
print(classification_report(y_true_eu, y_pred_eu,
      target_names=['absent(0)', 'present(1)'], zero_division=0))
print(confusion_matrix(y_true_eu, y_pred_eu))