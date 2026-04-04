# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ────────────────────────────────────────────────────────────────────
MANUAL_DATASET_FILE = Path(__file__).resolve().parents[3] / "data" / "large_coding_dataset.xlsx"
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 256       # max tokens per segment
BATCH_SIZE = 8         # small batch size for small dataset
EPOCHS = 4             # enough to fine-tune without overfitting
LEARNING_RATE = 2e-5   # standard for BERT fine-tuning
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── LOAD AND PREPARE DATA ─────────────────────────────────────────────────────
df = pd.read_excel(MANUAL_DATASET_FILE)

# Clean HTML and join sentences into single string per segment
def clean_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

def prepare_text(segment_string):
    try:
        segment_list = ast.literal_eval(segment_string)
    except Exception:
        segment_list = [str(segment_string)]
    cleaned = [clean_html_tags(s) for s in segment_list]
    return " ".join(cleaned)

df['text'] = df['context_sentences'].apply(prepare_text)

# Binary intensity
df['intensity'] = df['intensity'].apply(lambda x: 1 if x > 0 else 0)

# Filter noise if column exists
if 'issue distibution' in df.columns:
    df = df[df['issue distibution'] != 'noise'].reset_index(drop=True)

DIMENSIONS = ['intensity', 'eu_collective_framing', 'realist_stance', 'diplomatic_stance']

# ── DATASET CLASS ─────────────────────────────────────────────────────────────
class RhetoricalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── TRAINING FUNCTION ─────────────────────────────────────────────────────────
def train_bert_classifier(texts, labels, dimension_name):
    print(f"\n{'='*60}")
    print(f"Training BERT for: {dimension_name}")
    print(f"{'='*60}")

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=7, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=7, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    # Datasets and loaders
    train_dataset = RhetoricalDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = RhetoricalDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = RhetoricalDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model — num_labels=2 for binary classification
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Class weights for imbalance
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        [1.0 / c for c in class_counts], dtype=torch.float
    ).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # TRAINING LOOP
    best_val_f1 = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels_batch = batch['label'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch['label'].numpy())

        val_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

        # Saving best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"  → New best model saved (Val F1: {val_f1:.4f})")

    # TEST EVALUATION
    model.load_state_dict(best_model_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch['label'].numpy())

    test_f1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    print(f"\nTest F1 for {dimension_name}: {test_f1:.4f}")
    print(classification_report(test_true, test_preds,
          target_names=['absent(0)', 'present(1)'], zero_division=0))

    return model, tokenizer, test_f1

# TRAIN PER DIMENSION
bert_classifiers = {}
bert_results = {}
texts = df['text'].tolist()

for dimension in DIMENSIONS:
    # EU collective framing: France/Germany only
    if dimension == 'eu_collective_framing':
        mask = df['actor'].str.lower() != 'eeas'
        dim_texts = df[mask]['text'].tolist()
        dim_labels = df[mask][dimension].tolist()
    else:
        dim_texts = texts
        dim_labels = df[dimension].tolist()

    model, tokenizer, test_f1 = train_bert_classifier(
        dim_texts, dim_labels, dimension
    )
    bert_classifiers[dimension] = {'model': model, 'tokenizer': tokenizer}
    bert_results[dimension] = test_f1

# ── SAVE MODELS ───────────────────────────────────────────────────────────────
for dimension, clf_dict in bert_classifiers.items():
    save_path = Path(__file__).resolve().parents[3] / "data" / f"bert_{dimension}"
    clf_dict['model'].save_pretrained(save_path)
    clf_dict['tokenizer'].save_pretrained(save_path)
    print(f"Saved BERT model for {dimension} to {save_path}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n=== BERT FINE-TUNING RESULTS ===")
for dimension, f1 in bert_results.items():
    print(f"{dimension}: F1={f1:.4f}")