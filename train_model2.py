import os
import re
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score

KAGGLE_DATASET = "praveengovi/emotions-dataset-for-nlp" 
TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'
TEST_FILE = 'test.txt'

MODEL_DIR = 'saved_model'
MODEL_NAME = 'emotion_model'           
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
MODEL_ARCHIVE_PATH = MODEL_PATH + ".keras" 

VOCAB_SIZE = 20000      
MAX_LEN = 120
EMBEDDING_DIM = 100    
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
PATIENCE = 5

TOKENIZER_SAVE = os.path.join(MODEL_DIR, "vectorizer_vocab.pkl")
LABEL_ENCODER_SAVE = os.path.join(MODEL_DIR, "label_encoder.pkl")
H5_CHECKPOINT = os.path.join(MODEL_DIR, MODEL_NAME + ".h5") 

os.makedirs(MODEL_DIR, exist_ok=True)

def download_and_prepare_data():
    """
    Check for local files first. If not present, try to use kagglehub (if available),
    otherwise exit with a helpful message.
    """
    local_paths = [Path(TRAIN_FILE), Path(VAL_FILE), Path(TEST_FILE)]
    if all(p.exists() for p in local_paths):
        print("Found local train/val/test files.")
        return Path('.')

    
    try:
        import kagglehub
        print("Local files not found. Trying kagglehub dataset download...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        
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

def build_model(vocab_size, embedding_dim, max_len, num_classes):
    """
    Model with TextVectorization pre-applied outside (we include embedding layer here).
    Uses trainable embeddings learned from scratch.
    """
    inp = layers.Input(shape=(max_len,), dtype="int32", name="input_tokens")
    
    # Trainable embeddings from scratch
    emb_layer = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        trainable=True,  # Learn embeddings from data
        name="embedding"
    )(inp)

    # Architecture
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))(emb_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inp, outputs=out, name="emotion_bi_lstm")
    opt = optimizers.Adam(learning_rate=LR)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def get_predictions_from_dataset(model, dataset):
    """Get all predictions and true labels from a tf.data.Dataset"""
    all_predictions = []
    all_true_labels = []
    
    for batch in dataset:
        texts, true_labels = batch
        batch_predictions = model.predict(texts, verbose=0)
        all_predictions.extend(batch_predictions)
        all_true_labels.extend(true_labels.numpy())
    
    return np.array(all_predictions), np.array(all_true_labels)

def evaluate_model(model, train_data, val_data, test_data, y_train, y_val, y_test, label_encoder):
    """
    Comprehensive evaluation with confusion matrix and metrics for all datasets
    """
    results = {}
    
    datasets = [
        (train_data, y_train, 'train'),
        (val_data, y_val, 'val'), 
        (test_data, y_test, 'test')
    ]
    
    for dataset, true_labels, dataset_name in datasets:
       
        print(f"EVALUATION RESULTS FOR {dataset_name.upper()} SET")
        
        
        # Get predictions using the fixed function
        predictions, y_true = get_predictions_from_dataset(model, dataset)
        y_pred = np.argmax(predictions, axis=1)
        
        # Verify we have the right number of samples
        expected_samples = len(true_labels)
        actual_samples = len(y_true)
        
        if expected_samples != actual_samples:
            print(f" WARNING: Expected {expected_samples} samples but got {actual_samples}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Store results
        results[dataset_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        # Print results
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Samples:   {len(y_true)}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        class_names = label_encoder.classes_
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    return results

def save_evaluation_results(results, output_dir):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    metrics_dict = {}
    for dataset_name, result in results.items():
        metrics_dict[dataset_name] = {
            'accuracy': float(result['accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1_score': float(result['f1_score'])
        }
    
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Save confusion matrices
    for dataset_name, result in results.items():
        cm_path = os.path.join(output_dir, f'confusion_matrix_{dataset_name}.csv')
        np.savetxt(cm_path, result['confusion_matrix'], delimiter=',', fmt='%d')
    
    print(f"\nEvaluation results saved to: {output_dir}")

def main():
    print("Starting training pipeline with LEARNED embeddings (no GloVe)...")
    base_path = download_and_prepare_data()

    df_train = load_txt_df(base_path / TRAIN_FILE)
    df_val = load_txt_df(base_path / VAL_FILE)
    df_test = load_txt_df(base_path / TEST_FILE)

    if df_train.empty or df_val.empty or df_test.empty:
        raise RuntimeError("One or more data splits could not be loaded. Place train.txt / val.txt / test.txt in current directory.")

    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}")
    
    print("\nTraining class distribution:")
    print(df_train['emotion'].value_counts())
    print("\nValidation class distribution:")
    print(df_val['emotion'].value_counts())
    print("\nTest class distribution:")
    print(df_test['emotion'].value_counts())

    # Clean text
    df_train['text'] = df_train['text'].apply(clean_text_simple)
    df_val['text'] = df_val['text'].apply(clean_text_simple)
    df_test['text'] = df_test['text'].apply(clean_text_simple)

    
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['emotion'].values)
    y_val = le.transform(df_val['emotion'].values)
    y_test = le.transform(df_test['emotion'].values)

    num_classes = len(le.classes_)
    print("\nClasses:", le.classes_)

    vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_LEN,
        standardize=None  
    )
    vectorize_layer.adapt(df_train['text'].values)

    
    vocab = vectorize_layer.get_vocabulary()
    vocab_size = len(vocab)
    with open(TOKENIZER_SAVE, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved vectorizer vocab ({vocab_size} tokens) to {TOKENIZER_SAVE}")

    
    print("Using trainable embeddings learned from scratch (no pre-trained embeddings)")
    
    model = build_model(
        vocab_size=vocab_size, 
        embedding_dim=EMBEDDING_DIM, 
        max_len=MAX_LEN,
        num_classes=num_classes
    )
    model.summary()

    
    def make_ds(texts, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(texts), reshuffle_each_iteration=True)
        
        ds = ds.map(lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(df_train['text'].values, y_train, shuffle=True)
    val_ds = make_ds(df_val['text'].values, y_val, shuffle=False)
    test_ds = make_ds(df_test['text'].values, y_test, shuffle=False)

    
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw = {int(classes[i]): float(class_weights[i]) for i in range(len(classes))}
    print("Class weights:", cw)

    # Callbacks
    cbks = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE, 
            restore_best_weights=True, 
            verbose=1,
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5, 
            patience=3, 
            min_lr=1e-6, 
            verbose=1,
            mode='max'
        ),
        callbacks.ModelCheckpoint(
            H5_CHECKPOINT, 
            save_best_only=True, 
            monitor='val_accuracy',
            verbose=1,
            mode='max'
        )
    ]


    print(f"Training for up to {EPOCHS} epochs with early stopping patience {PATIENCE}")
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=cbks, 
        class_weight=cw, 
        verbose=1
    )

    print(f"Total epochs trained: {len(history.history['accuracy'])}")
    if len(history.history['accuracy']) > 1:
        print(f"Training Accuracy: {history.history['accuracy'][0]:.4f} → {history.history['accuracy'][-1]:.4f}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][0]:.4f} → {history.history['val_accuracy'][-1]:.4f}")
        print(f"Training Loss: {history.history['loss'][0]:.4f} → {history.history['loss'][-1]:.4f}")
        print(f"Validation Loss: {history.history['val_loss'][0]:.4f} → {history.history['val_loss'][-1]:.4f}")
    else:
        
        print("Training Accuracy:", history.history['accuracy'][0])
        print("Validation Accuracy:", history.history['val_accuracy'][0])
    
    with open(LABEL_ENCODER_SAVE, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved LabelEncoder to {LABEL_ENCODER_SAVE}")

    
    evaluation_results = evaluate_model(model, train_ds, val_ds, test_ds, y_train, y_val, y_test, le)
    
    save_evaluation_results(evaluation_results, os.path.join(MODEL_DIR, 'evaluation'))

    # Create inference pipeline
    string_input = layers.Input(shape=(1,), dtype=tf.string, name="raw_text")
    x = vectorize_layer(string_input)
    preds = model(x)
    pipeline_model = models.Model(inputs=string_input, outputs=preds, name="emotion_pipeline")

    # Clean up old models
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

    # Save the model
    try:
        pipeline_model.save(MODEL_ARCHIVE_PATH, include_optimizer=False)
        print(f"Saved full pipeline as single-file Keras archive: {MODEL_ARCHIVE_PATH}")
    except Exception as ex:
        print(f"ERROR: Failed to save .keras file: {ex}")
        raise

    history_path = os.path.join(MODEL_DIR, "history.json")
    with open(history_path, "w") as fh:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, fh)
    print("Saved training history to", history_path)

    print(f"Main inference artifact saved to: {MODEL_ARCHIVE_PATH}")
    print(f"Evaluation results saved to: {os.path.join(MODEL_DIR, 'evaluation')}")
    
    # Final performance summary
    train_acc = evaluation_results['train']['accuracy']
    val_acc = evaluation_results['val']['accuracy']
    test_acc = evaluation_results['test']['accuracy']
    
   
    print(f"Training Accuracy:   {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    

if __name__ == "__main__":
    main()


