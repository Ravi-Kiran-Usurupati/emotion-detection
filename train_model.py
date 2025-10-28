#!/usr/bin/env python3
"""
train_model.py

Improved training pipeline for emotion classification:
- TextVectorization (integrated into the model)
- Optional GloVe embeddings (auto-download)
- Bidirectional LSTM architecture
- tf.data pipeline, class weights, callbacks
- Saves a single Keras pipeline file (.keras) ready for inference.
"""

import os
import re
import json
import zipfile
import requests
import io
import pickle
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# -------- Config (tweak these) --------
KAGGLE_DATASET = "praveengovi/emotions-dataset-for-nlp"  # kept for reference
TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'
TEST_FILE = 'test.txt'

MODEL_DIR = 'saved_model'
MODEL_NAME = 'emotion_model'            # base name
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
MODEL_ARCHIVE_PATH = MODEL_PATH + ".keras" # Final pipeline file

VOCAB_SIZE = 20000      # increase vocab for better coverage
MAX_LEN = 120
EMBEDDING_DIM = 100     # if using GloVe use 100d or 200d etc.
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
PATIENCE = 4

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"  # will attempt to download automatically
USE_GLOVE = True        # set False to skip GloVe and train embeddings end-to-end
GLOVE_DIM = EMBEDDING_DIM
GLOVE_DIR = "glove"

TOKENIZER_SAVE = os.path.join(MODEL_DIR, "vectorizer_vocab.pkl")
LABEL_ENCODER_SAVE = os.path.join(MODEL_DIR, "label_encoder.pkl")
H5_CHECKPOINT = os.path.join(MODEL_DIR, MODEL_NAME + ".h5") # Legacy H5 checkpoint

# ---------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GLOVE_DIR, exist_ok=True)

# ---------- Utilities ----------
def download_and_prepare_data():
    """
    Check for local files first. If not present, try to use kagglehub (if available),
    otherwise exit with a helpful message.
    """
    local_paths = [Path(TRAIN_FILE), Path(VAL_FILE), Path(TEST_FILE)]
    if all(p.exists() for p in local_paths):
        print("Found local train/val/test files.")
        return Path('.')

    # If files aren't local, try to download with kagglehub if installed
    try:
        import kagglehub
        print("Local files not found. Trying kagglehub dataset download...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        # kagglehub usually returns an extraction directory - check expected files
        expected = [Path(path) / TRAIN_FILE, Path(path) / VAL_FILE, Path(path) / TEST_FILE]
        if all(p.exists() for p in expected):
            print("Downloaded dataset via kagglehub.")
            return Path(path)
        else:
            print("kagglehub download succeeded but expected files are missing in the archive.")
    except Exception as e:
        print("kagglehub not available or failed:", e)

    raise FileNotFoundError(f"Could not find {TRAIN_FILE}, {VAL_FILE}, {TEST_FILE} locally. Please place them in the current directory.")

def load_txt_df(path):
    """Load 'text;emotion' format txt into DataFrame with columns ['text','emotion']"""
    try:
        df = pd.read_csv(path, sep=';', header=None, names=['text', 'emotion'], engine='python')
        df['text'] = df['text'].astype(str)
        return df
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return pd.DataFrame(columns=['text', 'emotion'])

def clean_text_simple(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'http\S+', ' ', s)           # remove urls
    s = re.sub(r'[^a-zA-Z\s]', ' ', s, flags=re.I) # keep letters and spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------- GloVe loader ----------
def download_glove_if_needed(glove_dir=GLOVE_DIR, url=GLOVE_URL):
    """Download Stanford GloVe (glove.6B.zip) and extract glove.6B.{dim}d.txt"""
    zip_path = Path(glove_dir) / "glove.6B.zip"
    target_file = Path(glove_dir) / f"glove.6B.{GLOVE_DIM}d.txt"
    
    if target_file.exists():
        print("Found GloVe embeddings locally.")
        return target_file

    print("Downloading GloVe embeddings (this can be large ~800MB). If you don't want this, set USE_GLOVE=False.)")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as fw:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fw.write(chunk)
        print("Extracting GloVe...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            target = f"glove.6B.{GLOVE_DIM}d.txt"
            if target in z.namelist():
                z.extract(target, path=glove_dir)
                return target_file
            else:
                raise RuntimeError(f"{target} not found inside zip.")
    except Exception as e:
        print("Failed to download/extract GloVe:", e)
        return None
    finally:
        # Clean up the zip file after extraction
        if zip_path.exists():
            try:
                os.remove(zip_path)
            except Exception as e:
                print(f"Warning: could not remove zip file {zip_path}: {e}")


def build_embedding_matrix(vocab, glove_path, dim=GLOVE_DIM):
    """
    Create embedding matrix for the vocabulary using GloVe file at glove_path.
    vocab: list-of-terms where index in list corresponds to vector index in embedding layer.
    """
    print("Building embedding matrix using GloVe...")
    embeddings_index = {}
    glove_path = Path(glove_path)
    with glove_path.open("r", encoding="utf8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            try:
                nums = np.asarray(parts[1:], dtype='float32')
                if nums.shape[0] == dim: # Ensure dimension matches
                    embeddings_index[word] = nums
            except ValueError:
                continue # Skip lines that don't parse
                
    print(f"Loaded {len(embeddings_index)} glove vectors.")
    
    # TextVectorization vocabulary includes '' (padding) and '[UNK]'
    # vocab[0] is '', vocab[1] is '[UNK]'
    num_tokens = len(vocab)
    embedding_matrix = np.random.normal(size=(num_tokens, dim)).astype(np.float32) * 0.01
    
    hits = 0
    # Start from i=2 to skip padding and OOV tokens
    for i, word in enumerate(vocab):
        if i < 2: 
            continue
        vect = embeddings_index.get(word)
        if vect is not None:
            embedding_matrix[i] = vect
            hits += 1
            
    print(f"Matched {hits} of {num_tokens - 2} vocab tokens to GloVe.")
    return embedding_matrix

# ---------- Build model ----------
def build_model(vocab_size, embedding_dim, max_len, num_classes, embedding_matrix=None, trainable_emb=False):
    """
    Model with TextVectorization pre-applied outside (we include embedding layer here).
    If embedding_matrix is provided, we initialize weights and (optionally) freeze them.
    """
    inp = layers.Input(shape=(max_len,), dtype="int32", name="input_tokens")
    if embedding_matrix is not None:
        emb_layer = layers.Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=trainable_emb, # Freeze GloVe layers if trainable_emb=False
            name="embedding"
        )(inp)
    else:
        emb_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                     input_length=max_len, name="embedding")(inp)

    x = layers.SpatialDropout1D(0.2)(emb_layer)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inp, outputs=out, name="emotion_bi_lstm")
    opt = optimizers.Adam(learning_rate=LR)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

# ---------- Main ----------
def main():
    print("Starting improved training pipeline...")
    base_path = download_and_prepare_data()

    df_train = load_txt_df(base_path / TRAIN_FILE)
    df_val = load_txt_df(base_path / VAL_FILE)
    df_test = load_txt_df(base_path / TEST_FILE)

    if df_train.empty or df_val.empty or df_test.empty:
        raise RuntimeError("One or more data splits could not be loaded. Place train.txt / val.txt / test.txt in current directory.")

    # Clean text
    df_train['text'] = df_train['text'].apply(clean_text_simple)
    df_val['text'] = df_val['text'].apply(clean_text_simple)
    df_test['text'] = df_test['text'].apply(clean_text_simple)

    # Label encode
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['emotion'].values)
    y_val = le.transform(df_val['emotion'].values)
    y_test = le.transform(df_test['emotion'].values)

    num_classes = len(le.classes_)
    print("Classes:", le.classes_)

    # Build TextVectorization and adapt on training texts
    vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_LEN,
        standardize=None  # we've already cleaned text
    )
    vectorize_layer.adapt(df_train['text'].values)

    # Save vectorizer vocabulary
    vocab = vectorize_layer.get_vocabulary()
    with open(TOKENIZER_SAVE, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved vectorizer vocab ({len(vocab)} tokens) to {TOKENIZER_SAVE}")

    # Optionally download GloVe and build embedding matrix
    embedding_matrix = None
    if USE_GLOVE:
        glove_txt = download_glove_if_needed()
        if glove_txt:
            embedding_matrix = build_embedding_matrix(vocab, glove_txt, dim=GLOVE_DIM)
            print("Using GloVe embeddings.")
        else:
            print("GloVe not available — continuing with trainable embeddings.")

    # Build model (we use vocab length to determine embedding input dim if no glove)
    # The vocab size for Embedding layer is len(vocab)
    model = build_model(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN,
                        num_classes=num_classes, embedding_matrix=embedding_matrix,
                        trainable_emb=(embedding_matrix is None)) # Train embedding if not using GloVe
    model.summary()

    # Build tf.data pipelines (vectorize on the fly)
    def make_ds(texts, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(texts), reshuffle_each_iteration=True)
        # vectorize texts -> produce token ids
        ds = ds.map(lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(df_train['text'].values, y_train, shuffle=True)
    val_ds = make_ds(df_val['text'].values, y_val, shuffle=False)
    test_ds = make_ds(df_test['text'].values, y_test, shuffle=False)

    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw = {int(classes[i]): float(class_weights[i]) for i in range(len(classes))}
    print("Class weights:", cw)

    # Callbacks
    cbks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        # Save .h5 checkpoint as a fallback
        callbacks.ModelCheckpoint(H5_CHECKPOINT, save_best_only=True, monitor='val_loss', verbose=1)
    ]

    # Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbks, class_weight=cw, verbose=2)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

    # Save label encoder
    with open(LABEL_ENCODER_SAVE, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved LabelEncoder to {LABEL_ENCODER_SAVE}")

    # Save entire pipeline: we will wrap vectorize_layer + model into a new Model for inference and save it
    # Create a small Keras model that accepts raw string input, vectorizes, then applies trained model
    string_input = layers.Input(shape=(1,), dtype=tf.string, name="raw_text")
    x = vectorize_layer(string_input)
    preds = model(x)
    pipeline_model = models.Model(inputs=string_input, outputs=preds, name="emotion_pipeline")

    # --- FIX: Save as a single .keras file ---
    # This is the modern format Keras 3's load_model() prefers.
    
    # Remove old artifacts if they exist
    if os.path.isdir(MODEL_PATH):
        try:
            shutil.rmtree(MODEL_PATH)
            print(f"Removed existing model directory: {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: could not remove existing directory {MODEL_PATH}: {e}")
    
    if os.path.exists(MODEL_ARCHIVE_PATH):
        try:
            os.remove(MODEL_ARCHIVE_PATH)
            print(f"Removed existing .keras file: {MODEL_ARCHIVE_PATH}")
        except Exception as e:
            print(f"Warning: could not remove existing file {MODEL_ARCHIVE_PATH}: {e}")

    # Save the full pipeline as a single .keras file
    try:
        pipeline_model.save(MODEL_ARCHIVE_PATH, include_optimizer=False)
        print(f"Saved full pipeline as single-file Keras archive: {MODEL_ARCHIVE_PATH}")
    except Exception as ex:
        print(f"ERROR: Failed to save .keras file: {ex}")
        raise

    # Also save training history for debugging
    history_path = os.path.join(MODEL_DIR, "history.json")
    with open(history_path, "w") as fh:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, fh)
    print("Saved training history to", history_path)

    print("\nTraining complete.")
    print(f"Main inference artifact saved to: {MODEL_ARCHIVE_PATH}")
    print(f"Legacy H5 (no vectorizer) saved to: {H5_CHECKPOINT}")
    print(f"Label encoder saved to: {LABEL_ENCODER_SAVE}")
    print(f"Tokenizer vocab saved to: {TOKENIZER_SAVE}")


# (predict_texts function removed as it's not used in this script)

# ---------- Entry ----------
if __name__ == "__main__":
    main()

