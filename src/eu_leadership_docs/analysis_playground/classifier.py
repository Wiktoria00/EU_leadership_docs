# #imports
# import pandas as pd
# import logging
# import numpy as np
# import re
# from pathlib import Path
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# from sklearn.metrics import confusion_matrix, classification_report
# import ast
# import spacy
# from eu_leadership_docs.config import configure_logging
# nlp = spacy.load('en_core_web_sm')
# configure_logging()
# logger = logging.getLogger(__name__)

# # File paths
# MANUAL_DATASET_FILE = Path(__file__).resolve().parents[3] / "data" / "manual_coding_dataset.xlsx"

# # first i need to do feature matrix (rates not frequencies! since text length varies)
# manual_dataset_original = pd.read_excel(MANUAL_DATASET_FILE)
# manual_dataset = manual_dataset_original.copy()

# keyword_lists = {
#     'realist': ["sanction", "condemn", "violation", "annexation", "firmness", "illegal", 
#                 "unlawfully detain", "human right", "illegitimate","troop", "threat", "punishment", 
#                 "dependence", "gas", "energy", "strong signal", "resilient", "nord stream 2", "competitive", 
#                 "european army", "violence", "accountable", "restrictive measure", "restrictive", "defence", 
#                 "embargo", "arms", "tariffs", "disinformation", "obligation", "separatist", "impose", 
#                 "security", "destructive", "tough", "cyber attack", "repowereu", "military assistance", 
#                 "eumam", "training", "nato", "moldova", "oil", "civil protection mechanism", "military support", 
#                 "provocation", "shadow fleet"],
#     'diplomatic': ["dialogue", "minsk", "diplomatic", "peaceful", "de-escalation", "consultation", "partner", 
#                    "cessation", "humanitarian", "ease", "negotiate", "discussion", "meeting", "normandy", 
#                    "ceasefire", "cooperation", "solution", "common ground", "relationship", "agree", "engage", 
#                    "trilateral", "cooperation", "friend", "mediation", "counterpart", "discuss", "peace plan", 
#                    "delegation", "trilateral contact group", "mutual", "alliance", "common interest", "euren", 
#                    "constructive", "solidarity", "round table"],
#     'EU_collective':["european", "european family", "eumam", "neighbourhood policy", "eastern partnership", 
#                      "osce", "energy union", "odihr", "eastern partner", "eu engagement", "european crisis", 
#                      "council of europe", "28 Member State", "27 Member State", "european partner", "european council", 
#                      "european unification", "collective decision by the european union", "by the european union", 
#                      "european Union", "partners in the european union", "european decision-making", "european decision", 
#                      "european solidarity", "europe 's security"],
#     'intensity_moderate':["must", "territorial integrity", "commit to", "mobilize", "firmness", "require to", 
#                           "call on", "concern", "expect", "pressure", "necessary", "aggression","not have time",
#                           "stress the","underline","crucial","immediately","illegal","terrible",
#                           "urge","demand","unjustifiable","unacceptable","reiterate","serious","enforce","fail",
#                           "essential","great dismay","great concern","regret","disaster","setback","stress","worry",
#                           "cynical","attack","disregard","condemn","annexation", "reaffirm", "appeal","brutal","strongly condemn"],

#     'intensity_high':["extremely concern", "extremely worry","illegal annexation", "all mean", "human rights violation",
#                       "gravely concern","grave concern","cruel act","extremely worry","non-recognition policy",
#                       "murderous persistence","deplore","denounce","outrage","strongest condemnation",
#                       "underline concern","atrocity","strongest possible term",
#                       "war crime","immense suffer","full-scale invasion","flagrant violation",
#                       "unprovoke","ruthless","outrageous","heinous","blatant disregard"]
# }
# # move functions to helpers and import here
# def extract_features(segment_list, keyword_lists):
#     segment_text = " ".join(segment_list).lower()
#     doc = nlp(segment_text)
#     lemmatized_tokens = [token.lemma_ for token in doc]
#     lemmatized_text = ' '.join(lemmatized_tokens)
#     total_words = len(lemmatized_tokens)
#     if total_words == 0:
#         return {f"{category}_rate": 0 for category in keyword_lists.keys()}
#     features = {}
#     for category, keywords in keyword_lists.items():
#         keyword_count =  0
#         for keyword in keywords:
#             keyword_doc = nlp(keyword.lower())
#             lemmatized_keyword = ' '.join([token.lemma_ for token in keyword_doc])
#             pattern = r'\b' + re.escape(lemmatized_keyword) + r'\b'
#             matches = re.findall(pattern, lemmatized_text)
#             keyword_count += len(matches)
#         rate = keyword_count / total_words
#         features[f'{category}_rate'] = rate
#     return features

# def clean_html_tags(text):
#     cleaned = re.sub(r'<[^>]+>', '', text)
#     return cleaned

# def derive_intensity_from_rates(intensity_moderate_rate, intensity_high_rate):
#     if intensity_high_rate > 0.001:
#         return 2
#     elif intensity_moderate_rate > 0.00:
#         return 1
#     else:
#         return 0
# ######
# # DIAGNOSTIC FOR NOISE CAT
# # Check if your dataset has a category/noise column
# print(manual_dataset.columns.tolist())

# # If there's a noise indicator column, e.g. 'category'
# noise_mask = manual_dataset['issue distibution'] == 'noise'  # adjust column name
# print(f"\nNoise segments: {noise_mask.sum()} out of {len(manual_dataset)}")

# print("\nNoise label distribution:")
# print(manual_dataset[noise_mask][['intensity', 'eu_collective_framing', 
#                                    'realist_stance', 'diplomatic_stance']].value_counts())

# print("\nNon-noise label distribution for intensity:")
# print(manual_dataset[~noise_mask]['intensity'].value_counts())

# non_noise_mask = manual_dataset['issue distibution'] != 'noise'  # adjust column name
# manual_dataset_clean = manual_dataset[non_noise_mask].reset_index(drop=True)

# # Rebuild everything from manual_dataset_clean instead of manual_dataset
# sample_segments = manual_dataset_clean['translated_context_sentences']
# manual_codes = manual_dataset_clean[['intensity', 'eu_collective_framing', 
#                                       'realist_stance', 'diplomatic_stance']]
# #####
# #sample_segments = manual_dataset['translated_context_sentences']
# #manual_codes = manual_dataset[['intensity','eu_collective_framing','realist_stance','diplomatic_stance']]

# feature_matrix = []
# for segment_string in sample_segments:
#     segment_list = ast.literal_eval(segment_string)
#     segment_list_cleaned = [clean_html_tags(sentence) for sentence in segment_list]
#     features = extract_features(segment_list_cleaned, keyword_lists)
#     #adding to matrix
#     features_row = [
#         features['realist_rate'],
#         features['diplomatic_rate'],
#         features['EU_collective_rate'],
#         features['intensity_moderate_rate'],
#         features['intensity_high_rate']
#     ]
#     feature_matrix.append(features_row)

# X = np.array(feature_matrix)
# # test
# print(X.shape) # shuould be (341, 5)

# # Apply to all segments
# X_df = pd.DataFrame(X)
# X_df.columns = ['realist_rate', 'diplomatic_rate', 'EU_collective_rate', 
#                        'intensity_moderate_rate', 'intensity_high_rate']

# # Apply to all segments
# derived_intensity = X_df.apply(
#     lambda row: derive_intensity_from_rates(row['intensity_moderate_rate'], row['intensity_high_rate']),
#     axis=1
# )

# # # Validate: does derived intensity match your manual codes?
# # manual_intensity = np.array(manual_codes['intensity'])

# # agreement = (derived_intensity == manual_intensity).mean()
# # print(f"Agreement between derived and manual intensity: {agreement:.1%}")

# # # See where they disagree
# # disagreements = derived_intensity != manual_intensity
# # print(f"\nDisagreements: {disagreements.sum()} out of {len(derived_intensity)}")

# # disagreement_idx = np.where(disagreements)[0]

# # print("Examples where derived != manual:")
# # for idx in disagreement_idx[:10]:
# #     print(f"\nSegment {idx}:")
# #     print(f"  Derived: {derived_intensity[idx]}, Manual: {manual_intensity[idx]}")
# #     print(f"  Moderate rate: {X_df.iloc[idx]['intensity_moderate_rate']:.4f}")
# #     print(f"  High rate: {X_df.iloc[idx]['intensity_high_rate']:.4f}")
# #     print(f"  Segment: {sample_segments[idx][:150]}...")
# # #####

# # Define relevant features per dimension

# feature_cols = {
#     'intensity': ['intensity_moderate_rate', 'intensity_high_rate'],
#     'eu_collective_framing': ['EU_collective_rate'],
#     'realist_stance': ['realist_rate'],
#     'diplomatic_stance': ['diplomatic_rate']
# }

# classifiers = {}
# results = {}
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1.0, 10.0],
#     'solver': ['lbfgs'],
#     'max_iter': [1000, 2000]
# }

# for dimension in ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']:
#     logger.info(f"Training the classifier for {dimension}")

#     # Use only relevant features for this dimension
#     cols = feature_cols[dimension]
#     X_dim = X_df[cols].values

#     y = np.array(manual_codes[dimension])
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_dim, y, test_size=0.2, random_state=7, stratify=y
#     )

#     grid_search = GridSearchCV(
#         estimator=LogisticRegression(random_state=7, class_weight='balanced'),
#         param_grid=param_grid,
#         cv=5,
#         scoring='f1_weighted',  # better than accuracy for imbalanced classes
#         verbose=1
#     )
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     logger.info(f"Best parameters for {dimension}: {best_params}")

#     clf = grid_search.best_estimator_
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_test, y_pred, average='weighted', zero_division=0
#     )
#     logger.info(f"Performance for {dimension} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

#     classifiers[dimension] = clf
#     results[dimension] = {
#         'best_params': best_params,
#         'best_cv_score': grid_search.best_score_,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'feature_cols': cols  # store so validation loop knows which features to use
#     }
# print(manual_codes['intensity'].value_counts())


# ###### we now valuate on the full coded sample dataset :)
# logger.info("Validating classifiers on the full coded sample dataset...")
# validation_results = {}
# for dimension in ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']:
#     clf = classifiers[dimension]
#     cols = feature_cols[dimension]
#     X_dim = X_df[cols].values

#     y = np.array(manual_codes[dimension])
#     y_pred = clf.predict(X_dim)  # use X_dim, not X
#     accuracy = accuracy_score(y, y_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y, y_pred, average='weighted', zero_division=0
#     )
#     validation_results[dimension] = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1
#     }
# print("\nValidation results on the full coded sample dataset:")
# for dimension, metrics in validation_results.items():
#     print(f"{dimension.capitalize()}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1_score']:.4f}")

# y_true = np.array(manual_codes['intensity'])
# y_rule = np.array([
#     derive_intensity_from_rates(row['intensity_moderate_rate'], row['intensity_high_rate'])
#     for _, row in X_df.iterrows()
# ])

# # Focus only on moderate→high errors
# print("=== moderate(true=1) predicted as high(2) ===")
# for idx in range(len(y_true)):
#     if y_true[idx] == 1 and y_rule[idx] == 2:
#         segment_text = " ".join(ast.literal_eval(sample_segments.iloc[idx]))
        
#         # Show which high keywords matched
#         doc = nlp(segment_text.lower())
#         lemmatized_text = ' '.join([token.lemma_ for token in doc])
#         matched_high = []
#         for keyword in keyword_lists['intensity_high']:
#             kw_doc = nlp(keyword.lower())
#             lemmatized_kw = ' '.join([token.lemma_ for token in kw_doc])
#             pattern = r'\b' + re.escape(lemmatized_kw) + r'\b'
#             if re.search(pattern, lemmatized_text):
#                 matched_high.append(keyword)
        
#         print(f"\nSegment {idx}: high_rate={X_df.iloc[idx]['intensity_high_rate']:.4f}")
#         print(f"  Matched high keywords: {matched_high}")
#         print(f"  Text: {segment_text[:300]}...")

# acc = accuracy_score(y_true, y_rule)
# precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_rule, average='weighted', zero_division=0)
# print(f"\nRule-based intensity: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# ## test to see where classifier fails
# y_true_intensity = np.array(manual_codes['intensity'])

# # Compare both approaches side by side
# y_pred_classifier = classifiers['intensity'].predict(X_df[feature_cols['intensity']].values)
# y_pred_rulebased = np.array([
#     derive_intensity_from_rates(row['intensity_moderate_rate'], row['intensity_high_rate'])
#     for _, row in X_df.iterrows()
# ])

# print("=== CLASSIFIER ===")
# print(classification_report(y_true_intensity, y_pred_classifier, 
#       target_names=['none(0)', 'moderate(1)', 'high(2)'], zero_division=0))
# print(confusion_matrix(y_true_intensity, y_pred_classifier))

# print("\n=== RULE-BASED ===")
# print(classification_report(y_true_intensity, y_pred_rulebased,
#       target_names=['none(0)', 'moderate(1)', 'high(2)'], zero_division=0))
# print(confusion_matrix(y_true_intensity, y_pred_rulebased))

# ### NOISE SNIPPET


# #later we will scale to the full dataset, now i test it on the sample dataset to see if it works and if the results are reasonable.
# ### TESTING
# # print("\nDetailed disagreement analysis per dimension:")
# # for dimension in ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']:
# #     clf = classifiers[dimension]
# #     cols = feature_cols[dimension]
# #     X_dim = X_df[cols].values

# #     y_true = np.array(manual_codes[dimension])
# #     y_pred = clf.predict(X_dim)
    
# #     disagreements = y_true != y_pred
# #     disagreement_idx = np.where(disagreements)[0]
    
# #     print(f"\n{'='*60}")
# #     print(f"{dimension.upper()} — {disagreements.sum()} disagreements out of {len(y_true)} ({disagreements.mean():.1%})")
# #     print(f"{'='*60}")
    
# #     # Class distribution of disagreements
# #     print(f"\nTrue label distribution in disagreements:")
# #     for val, count in zip(*np.unique(y_true[disagreement_idx], return_counts=True)):
# #         print(f"  Label {val}: {count} times")
    
# #     print(f"\nPredicted label distribution in disagreements:")
# #     for val, count in zip(*np.unique(y_pred[disagreement_idx], return_counts=True)):
# #         print(f"  Label {val}: {count} times")

# #     # Confusion-style breakdown: what did the model predict when it was wrong?
# #     print(f"\nTrue → Predicted (in disagreements):")
# #     for idx in disagreement_idx:
# #         true_val = y_true[idx]
# #         pred_val = y_pred[idx]
# #         print(f"  Segment {idx:>3}: true={true_val}, pred={pred_val}  | "
# #               f"realist={X_df.iloc[idx]['realist_rate']:.4f}, "
# #               f"diplomatic={X_df.iloc[idx]['diplomatic_rate']:.4f}, "
# #               f"EU_collective={X_df.iloc[idx]['EU_collective_rate']:.4f}, "
# #               f"mod={X_df.iloc[idx]['intensity_moderate_rate']:.4f}, "
# #               f"high={X_df.iloc[idx]['intensity_high_rate']:.4f}")


