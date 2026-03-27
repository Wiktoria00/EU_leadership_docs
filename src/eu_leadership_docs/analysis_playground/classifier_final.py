#imports
import pandas as pd
import logging
import numpy as np
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report
import ast
import spacy
from eu_leadership_docs.config import configure_logging
nlp = spacy.load('en_core_web_sm')
configure_logging()
logger = logging.getLogger(__name__)

# file paths
MANUAL_DATASET_FILE = Path(__file__).resolve().parents[3] / "data" / "manual_coding_dataset.xlsx"

manual_dataset_original = pd.read_excel(MANUAL_DATASET_FILE)
manual_dataset = manual_dataset_original.copy()

# keyword lists for feature extraction
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
    'eu_collective':["european family", "we european","european level", "at eu level", "european project","eumam", "neighbourhood policy", "eastern partnership", 
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

# i need to filter the noise column
non_noise_mask = manual_dataset['issue distibution'] != 'noise'
manual_dataset_clean = manual_dataset[non_noise_mask].reset_index(drop=True)
#checking the distribution of actors in the cleaned dataset
#print(manual_dataset.columns.tolist())
#print(manual_dataset_clean['actor'].value_counts())

# rebuilding sample segments and manual codes
sample_segments = manual_dataset_clean['translated_context_sentences']
manual_codes = manual_dataset_clean[['intensity', 'eu_collective_framing', 
                                      'realist_stance', 'diplomatic_stance']].copy()
manual_codes['intensity'] = manual_codes['intensity'].apply(lambda x: 1 if x > 0 else 0)

# create Fr/Ger only mask for EU collective framing
non_eeas_mask = manual_dataset_clean['actor'] != 'eeas'
X_df_non_eeas_idx = manual_dataset_clean[non_eeas_mask].index
sample_segments_non_eeas = sample_segments[non_eeas_mask].reset_index(drop=True)
manual_codes_non_eeas = manual_codes[non_eeas_mask].reset_index(drop=True)

# feature matrix for ALL segments!
feature_matrix = []
for segment_string in sample_segments:
    segment_list = ast.literal_eval(segment_string)
    segment_list_cleaned = [clean_html_tags(sentence) for sentence in segment_list]
    features = extract_features(segment_list_cleaned, keyword_lists)
    features_row = [
        features['realist_rate'],
        features['diplomatic_rate'],
        features['eu_collective_rate'],
        features['intensity_rate']
    ]
    feature_matrix.append(features_row)

X = np.array(feature_matrix)

X_df = pd.DataFrame(X)
X_df.columns = ['realist_rate', 'diplomatic_rate', 'eu_collective_rate', 'intensity_rate']

# feature matrix for Fr/Ger only (for EU collective)
X_df_non_eeas = X_df[non_eeas_mask].reset_index(drop=True)

feature_cols = {
    'intensity': ['intensity_rate'],
    'eu_collective_framing': ['eu_collective_rate'],
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

    # EU collective framing, train on Fr/Ger only
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
    print(f"Performance for {dimension} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

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

# validation on full coded sample
logger.info("Validating classifiers on the full coded sample dataset...")
validation_results = {}
for dimension in ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']:
    clf = classifiers[dimension]
    cols = feature_cols[dimension]

    #EU collective framing: validate on Fr/Ger only
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

# print("\nValidation results on the full coded sample dataset:")
# for dimension, metrics in validation_results.items():
#     print(f"{dimension.capitalize()}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1_score']:.4f}")

# evaluation of unseen data!!!

# File path
UNSEEN_DATASET_FILE = Path(__file__).resolve().parents[3] / "data" / "unseen_data_dataset.xlsx"
unseen_df = pd.read_excel(UNSEEN_DATASET_FILE)

# manual codes 
unseen_manual_codes = unseen_df[['intensity', 'realist_stance', 
                                  'diplomatic_stance', 'eu_collective_framing']].copy()
unseen_manual_codes['intensity'] = unseen_manual_codes['intensity'].apply(
    lambda x: 1 if x > 0 else 0
)

# feature matrix
unseen_feature_matrix = []
for segment_string in unseen_df['context_sentences']:
    try:
        segment_list = ast.literal_eval(segment_string)
    except Exception:
        segment_list = [str(segment_string)]
    segment_list_cleaned = [clean_html_tags(sentence) for sentence in segment_list]
    features = extract_features(segment_list_cleaned, keyword_lists)
    features_row = [
        features['realist_rate'],
        features['diplomatic_rate'],
        features['eu_collective_rate'],
        features['intensity_rate']
    ]
    unseen_feature_matrix.append(features_row)

X_unseen = np.array(unseen_feature_matrix)
X_unseen_df = pd.DataFrame(X_unseen)
X_unseen_df.columns = ['realist_rate', 'diplomatic_rate', 
                        'eu_collective_rate', 'intensity_rate']

# EEAS mask for EU collective framing 
non_eeas_mask_unseen = unseen_df['actor'] != 'eeas'
X_unseen_non_eeas = X_unseen_df[non_eeas_mask_unseen].reset_index(drop=True)
manual_codes_unseen_non_eeas = unseen_manual_codes[non_eeas_mask_unseen].reset_index(drop=True)

# classifiers on unseen data and comparison to manual codes
print("\nCLASSIFIER PERFORMANCE ON UNSEEN DATA")
unseen_results = {}

for dimension in ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']:
    clf = classifiers[dimension]
    cols = feature_cols[dimension]

    if dimension == 'eu_collective_framing':
        X_dim = X_unseen_non_eeas[cols].values
        y_true = np.array(manual_codes_unseen_non_eeas[dimension])
    else:
        X_dim = X_unseen_df[cols].values
        y_true = np.array(unseen_manual_codes[dimension])

    y_pred = clf.predict(X_dim)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    unseen_results[dimension] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"\n--- {dimension.upper()} ---")
    print(classification_report(y_true, y_pred,
          target_names=['absent(0)', 'present(1)'], zero_division=0))

# comparison in a few lines: validated vs unseen
print("\nSUMMARY: VALIDATED vs UNSEEN PERFORMANCE")
print(f"{'Dimension':<25} {'Validated F1':>12} {'Unseen F1':>10} {'Difference':>12}")
print("-" * 62)
for dimension in ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']:
    validated_f1 = validation_results[dimension]['f1_score']
    unseen_f1 = unseen_results[dimension]['f1_score']
    diff = unseen_f1 - validated_f1
    print(f"{dimension:<25} {validated_f1:>12.4f} {unseen_f1:>10.4f} {diff:>+12.4f}")

# #comparing to rule-based intensity
# y_true = np.array(manual_codes['intensity'])
# y_rule = np.array([
#     derive_intensity_from_rates(row['intensity_rate'])
#     for _, row in X_df.iterrows()
# ])

# acc = accuracy_score(y_true, y_rule)
# precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_rule, average='weighted', zero_division=0)
# print(f"\nRule-based intensity: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


# showing final output: 
  
# # EU collective framing detailed validation
# eu_present = manual_codes_non_eeas['eu_collective_framing'] == 1
# eu_zero_rate = X_df_non_eeas['EU_collective_rate'] == 0

# print("\nEU Collective Framing Coverage (France/Germany only):")
# print(f"  Coded present AND rate > 0: {(eu_present & ~eu_zero_rate).sum()}")
# print(f"  Coded present BUT rate = 0: {(eu_present & eu_zero_rate).sum()}")
# print(f"  Coded absent AND rate > 0: {(~eu_present & ~eu_zero_rate).sum()}")
# print(f"  Coded absent AND rate = 0: {(~eu_present & eu_zero_rate).sum()}")
# print(f"\nLabel distribution: {dict(manual_codes_non_eeas['eu_collective_framing'].value_counts())}")

# clf = classifiers['eu_collective_framing']
# X_dim = X_df_non_eeas[feature_cols['eu_collective_framing']].values
# y_true_eu = np.array(manual_codes_non_eeas['eu_collective_framing'])
# y_pred_eu = clf.predict(X_dim)

# print("\n=== EU COLLECTIVE FRAMING (France/Germany only) ===")
# print(classification_report(y_true_eu, y_pred_eu,
#       target_names=['absent(0)', 'present(1)'], zero_division=0))
# print(confusion_matrix(y_true_eu, y_pred_eu))