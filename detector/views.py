# import json
# import os
# import re
# import pickle
# import numpy as np
# from django.http import JsonResponse
# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from django.conf import settings # Import settings

# # --- Load Model and Artifacts ---

# # Construct paths using Django's settings.BASE_DIR
# # Assumes 'saved_model' is in the project root
# MODEL_DIR = os.path.join(settings.BASE_DIR, 'saved_model')
# MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_model.h5')
# TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.json')
# ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
# MAX_LEN = 100 # This must match the 'maxlen' from your training script

# # Load artifacts at server startup
# try:
#     print("Loading model and artifacts...")
#     model = load_model(MODEL_PATH)
    
#     with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
#         tokenizer = tokenizer_from_json(json.load(f))
        
#     with open(ENCODER_PATH, 'rb') as f:
#         le = pickle.load(f)
    
#     print("Model and artifacts loaded successfully.")

# except FileNotFoundError as e:
#     print(f"--- CRITICAL ERROR: Model file not found ---")
#     print(f"Error: {e}")
#     print("Please run 'python train_model.py' to generate model files.")
#     model = None
#     tokenizer = None
#     le = None
# except Exception as e:
#     print(f"Error loading model artifacts: {e}")
#     model = None
#     tokenizer = None
#     le = None

# # --- Helper Function ---

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z\s']", '', text) # Keep letters, spaces, and apostrophes
#     return text

# # --- Views ---

# def index(request):
#     """Renders the main HTML page."""
#     return render(request, 'detector/index.html')

# @csrf_exempt # Use this decorator for simplicity, or handle CSRF properly
# def predict_emotion(request):
#     """Handles the API request for emotion prediction."""
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Invalid request method'}, status=405)
        
#     if not model or not tokenizer or not le:
#         return JsonResponse({'error': 'Model is not loaded'}, status=500)

#     try:
#         data = json.loads(request.body)
#         text = data.get('text', '')

#         if not text:
#             return JsonResponse({'error': 'No text provided'}, status=400)

#         # 1. Preprocess the input text
#         cleaned_text = clean_text(text)
#         sequence = tokenizer.texts_to_sequences([cleaned_text])
#         padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

#         # 2. Make prediction
#         pred = model.predict(padded_sequence)
        
#         # 3. Get all probabilities
#         probabilities = pred[0]
#         emotion_labels = le.classes_
        
#         # 4. Create the full predictions list
#         predictions = []
#         for label, prob in zip(emotion_labels, probabilities):
#             predictions.append({
#                 "emotion": label,
#                 "percentage": float(prob) * 100  # Convert to percentage
#             })
            
#         # 5. Sort by percentage (highest first)
#         predictions.sort(key=lambda x: x['percentage'], reverse=True)

#         # 6. Return the full list
#         return JsonResponse({"predictions": predictions})

#     except json.JSONDecodeError:
#         return JsonResponse({'error': 'Invalid JSON'}, status=400)
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

import json
import os
import re
import pickle
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# We need TextVectorization to recreate the layer for the legacy H5 path
from tensorflow.keras.layers import TextVectorization

# --- Paths & constants ---
MODEL_DIR = os.path.join(settings.BASE_DIR, 'saved_model')

# New pipeline artifacts (Keras 3 export or .keras)
PIPELINE_DIR = os.path.join(MODEL_DIR, 'emotion_model')          # saved_model/emotion_model (directory)
PIPELINE_ARCHIVE = PIPELINE_DIR + '.keras'                      # saved_model/emotion_model.keras

# Legacy artifacts (kept for fallback compatibility)
LEGACY_H5 = os.path.join(MODEL_DIR, 'emotion_model.h5')          # older single-file model
VECTOR_VOCAB_PKL = os.path.join(MODEL_DIR, 'vectorizer_vocab.pkl') # vocab from TextVectorization (from new train script)
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')  # label encoder

# This MUST match the MAX_LEN from the training script
MAX_LEN = 120

# --- Globals to hold loaded artifacts ---
model = None
label_encoder = None
vectorizer_layer_legacy = None # This will hold the recreated TextVectorization layer for the H5 fallback
using_pipeline = False
using_legacy_h5 = False

# --- Helper: clean text (MUST MATCH train_model.py's clean_text_simple) ---
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)           # remove urls
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, flags=re.I) # keep letters and spaces (removes apostrophes)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load model & artifacts at startup ---
print("Detector: loading model artifacts...")
try:
    # 1) Try to load pipeline SavedModel directory (preferred)
    if os.path.isdir(PIPELINE_DIR):
        print(f"Attempting to load pipeline SavedModel from directory: {PIPELINE_DIR}")
        model = tf.keras.models.load_model(PIPELINE_DIR)
        using_pipeline = True
        print("Loaded pipeline SavedModel (directory).")
    # 2) Try to load single-file .keras archive
    elif os.path.exists(PIPELINE_ARCHIVE):
        print(f"Attempting to load pipeline .keras archive: {PIPELINE_ARCHIVE}")
        model = tf.keras.models.load_model(PIPELINE_ARCHIVE)
        using_pipeline = True
        print("Loaded pipeline .keras archive.")
    # 3) Fallback to legacy .h5 (numeric-input model)
    elif os.path.exists(LEGACY_H5):
        print(f"Attempting to load legacy H5 model: {LEGACY_H5}")
        model = load_model(LEGACY_H5)
        using_legacy_h5 = True
        print("Loaded legacy H5 model.")
        
        # FIXED: Load the vectorizer_vocab.pkl, NOT tokenizer.json
        if os.path.exists(VECTOR_VOCAB_PKL):
            try:
                with open(VECTOR_VOCAB_PKL, 'rb') as f:
                    vocab = pickle.load(f)
                print(f"Loaded legacy vectorizer vocab ({len(vocab)} tokens).")
                
                # Recreate the layer *exactly* as in training
                vectorizer_layer_legacy = TextVectorization(
                    max_tokens=len(vocab), # Use loaded vocab size
                    output_mode='int',
                    output_sequence_length=MAX_LEN,
                    standardize=None, # We clean manually
                    vocabulary=vocab  # Set the loaded vocabulary
                )
                print("Recreated TextVectorization layer for legacy H5 model.")
            except Exception as e:
                print(f"Warning: failed to load/recreate legacy vectorizer from {VECTOR_VOCAB_PKL}: {e}")
        else:
            print(f"CRITICAL: Legacy H5 model loaded, but {VECTOR_VOCAB_PKL} not found. Fallback will fail.")
    else:
        print("No model artifact found in saved_model/ — please run your training script to produce artifacts.")
        model = None

    # Load LabelEncoder if present (used for mapping indices -> class names)
    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Loaded LabelEncoder.")
    else:
        print("LabelEncoder not found at", LABEL_ENCODER_PATH)
        label_encoder = None

except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model = None
    label_encoder = None
    vectorizer_layer_legacy = None
    using_pipeline = False
    using_legacy_h5 = False

# --- Views ---
def index(request):
    """Renders the main HTML page."""
    return render(request, 'detector/index.html')

@csrf_exempt
def predict_emotion(request):
    """API: POST JSON {'text': '...'} -> returns sorted list of predictions."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    if model is None or label_encoder is None:
        return JsonResponse({'error': 'Model or label encoder not loaded. Check server logs.'}, status=500)

    try:
        payload = json.loads(request.body)
        text = payload.get('text', '')
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)

        # Clean the text (matches the cleaning used during training)
        cleaned = clean_text(text)

        # --- Pipeline mode: model expects raw string input ---
        if using_pipeline:
            # Keras pipeline expects a batch of strings (e.g., tf.constant(["text"]))
            inp = tf.constant([cleaned])
            probs = model.predict(inp, verbose=0)
            probs = np.asarray(probs).squeeze()
            class_names = label_encoder.classes_

        # --- Legacy H5 mode: must use recreated vectorizer layer ---
        elif using_legacy_h5:
            if vectorizer_layer_legacy is None:
                return JsonResponse({'error': 'Legacy H5 model is loaded, but its vectorizer is missing.'}, status=500)
            
            # FIXED: Use the recreated vectorizer layer to preprocess the text
            inp_tensor = vectorizer_layer_legacy(tf.constant([cleaned]))
            probs = model.predict(inp_tensor, verbose=0)
            probs = np.asarray(probs).squeeze()
            class_names = label_encoder.classes_
            
        else:
            return JsonResponse({'error': 'No compatible model loaded.'}, status=500)

        # Build predictions list
        predictions = []
        for label, p in zip(class_names, probs):
            predictions.append({
                "emotion": str(label),
                "percentage": float(p) * 100.0
            })

        predictions.sort(key=lambda x: x['percentage'], reverse=True)
        return JsonResponse({"predictions": predictions})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        # log error server-side
        print(f"Prediction error: {e}")
        return JsonResponse({'error': f'An internal error occurred: {str(e)}'}, status=500)

