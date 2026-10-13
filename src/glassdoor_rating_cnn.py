"""
Glassdoor Review Rating Prediction
----------------------------------
Predicts a Glassdoor review's star rating (1-5) from its pros, cons and
headline text, using a hybrid CNN (text) + dense (engineered features)
neural network.

Final competition run: 100K-review training set, CNN model.
(XGBoost / LightGBM were tested but disabled to fit in memory.)

Data is not included (course dataset). Put the CSVs in data/ - see data/README.md.

Usage:
    python src/glassdoor_rating_cnn.py
"""

# CRITICAL: Set environment variables BEFORE importing TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import re
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports - AFTER setting environment variables
import tensorflow as tf

# Limit memory growth
physical_devices = tf.config.list_physical_devices('CPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     Input, Embedding, LSTM, GlobalMaxPooling1D,
                                     Concatenate, Conv1D, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Gradient Boosting
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not installed. pip install lightgbm")

# NLP libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not installed. pip install textblob")

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not installed. pip install nltk")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("=" * 70)
print("ECON 424 - ADVANCED PREDICTION MODEL")
print("Deep Learning + Ensemble + 50+ Engineered Features")
print("=" * 70)

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================
print("\n[1/10] LOADING DATA...")

# Load training data
# Option 1: Small only (100k rows - safest, uses less memory) - RECOMMENDED FOR 8GB RAM
# Option 2: Large only (500k rows - better accuracy, needs 12GB+ RAM)
# Option 3: Combined (600k rows - best accuracy, needs 16GB+ RAM) - USE IN GOOGLE COLAB

# Using SMALL dataset (100k rows) - Requested by User
train_df = pd.read_csv(os.path.join(DATA_DIR, "small_train.csv"))
print(f"  Using training set: {train_df.shape}")

# Load test data
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_without_response_v1.csv"))
print(f"  Test set: {test_df.shape}")

# ============================================================================
# SECTION 2: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[2/10] ADVANCED FEATURE ENGINEERING...")

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def count_specific_patterns(text):
    """Count specific linguistic patterns"""
    text = str(text)
    features = {}
    
    # Punctuation patterns
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['period_count'] = text.count('.')
    features['comma_count'] = text.count(',')
    
    # Capitalization patterns
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    features['all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
    
    # Number patterns
    features['number_count'] = len(re.findall(r'\d+', text))
    features['has_percentage'] = int('%' in text or 'percent' in text.lower())
    features['has_dollar'] = int('$' in text or 'dollar' in text.lower())
    
    return features

def compute_readability(text):
    """Compute readability metrics"""
    text = str(text)
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    sentences = max(sentences, 1)
    
    features = {}
    
    # Average word length
    if words:
        features['avg_word_length'] = np.mean([len(w) for w in words])
        features['max_word_length'] = max([len(w) for w in words])
        features['long_word_ratio'] = sum(1 for w in words if len(w) > 6) / len(words)
    else:
        features['avg_word_length'] = 0
        features['max_word_length'] = 0
        features['long_word_ratio'] = 0
    
    # Words per sentence
    features['words_per_sentence'] = len(words) / sentences
    
    # Syllable approximation (simple method)
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiou'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)
    
    if words:
        syllables = sum(count_syllables(w) for w in words)
        # Flesch Reading Ease approximation
        features['flesch_score'] = 206.835 - 1.015 * (len(words)/sentences) - 84.6 * (syllables/len(words))
        features['avg_syllables'] = syllables / len(words)
    else:
        features['flesch_score'] = 0
        features['avg_syllables'] = 0
    
    return features

def get_sentiment_features(text):
    """Extract sentiment features using TextBlob"""
    features = {}
    if TEXTBLOB_AVAILABLE and text:
        try:
            blob = TextBlob(str(text))
            features['polarity'] = blob.sentiment.polarity
            features['subjectivity'] = blob.sentiment.subjectivity
        except:
            features['polarity'] = 0
            features['subjectivity'] = 0
    else:
        features['polarity'] = 0
        features['subjectivity'] = 0
    return features

def get_pos_features(text):
    """Part-of-speech tagging features"""
    features = {}
    if NLTK_AVAILABLE and text:
        try:
            words = word_tokenize(str(text))[:100]  # Limit for speed
            pos_tags = nltk.pos_tag(words)
            
            # Count POS tags
            noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
            verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
            adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
            adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
            
            total = len(pos_tags) + 1
            features['noun_ratio'] = noun_count / total
            features['verb_ratio'] = verb_count / total
            features['adj_ratio'] = adj_count / total
            features['adv_ratio'] = adv_count / total
        except:
            features['noun_ratio'] = 0
            features['verb_ratio'] = 0
            features['adj_ratio'] = 0
            features['adv_ratio'] = 0
    else:
        features['noun_ratio'] = 0
        features['verb_ratio'] = 0
        features['adj_ratio'] = 0
        features['adv_ratio'] = 0
    return features

def count_keywords(text):
    """Count domain-specific keywords"""
    text = str(text).lower()
    
    positive_keywords = [
        'great', 'excellent', 'amazing', 'good', 'best', 'love', 'awesome',
        'fantastic', 'wonderful', 'perfect', 'flexible', 'supportive', 
        'growth', 'opportunity', 'balance', 'culture', 'friendly', 'benefits',
        'learning', 'development', 'team', 'collaborative', 'innovative'
    ]
    
    negative_keywords = [
        'bad', 'poor', 'terrible', 'worst', 'horrible', 'awful', 'hate',
        'toxic', 'stressful', 'underpaid', 'overworked', 'micromanage',
        'no growth', 'low pay', 'long hours', 'turnover', 'politics',
        'unfair', 'disorganized', 'chaotic', 'layoff', 'fired'
    ]
    
    neutral_keywords = [
        'okay', 'average', 'decent', 'normal', 'standard', 'typical',
        'nothing special', 'mediocre'
    ]
    
    features = {}
    features['positive_keyword_count'] = sum(1 for kw in positive_keywords if kw in text)
    features['negative_keyword_count'] = sum(1 for kw in negative_keywords if kw in text)
    features['neutral_keyword_count'] = sum(1 for kw in neutral_keywords if kw in text)
    features['keyword_sentiment_ratio'] = (features['positive_keyword_count'] + 1) / (features['negative_keyword_count'] + 1)
    
    return features

def engineer_features(df):
    """Comprehensive feature engineering"""
    df = df.copy()
    print("    - Cleaning text...")
    
    # Fill missing values
    df['headline'] = df['headline'].fillna("")
    df['cons'] = df['cons'].fillna("")
    df['pros'] = df['pros'].fillna("")
    df['job_title'] = df['job_title'].fillna("Unknown")
    df['firm'] = df['firm'].fillna("Unknown")
    
    # Clean text
    df['pros_clean'] = df['pros'].apply(clean_text)
    df['cons_clean'] = df['cons'].apply(clean_text)
    df['headline_clean'] = df['headline'].apply(clean_text)
    df['all_text'] = df['pros_clean'] + " " + df['cons_clean'] + " " + df['headline_clean']
    
    print("    - Basic text statistics...")
    # Basic text statistics
    df['pros_char_length'] = df['pros'].apply(lambda x: len(str(x)))
    df['cons_char_length'] = df['cons'].apply(lambda x: len(str(x)))
    df['headline_char_length'] = df['headline'].apply(lambda x: len(str(x)))
    df['total_char_length'] = df['pros_char_length'] + df['cons_char_length'] + df['headline_char_length']
    
    df['pros_word_count'] = df['pros_clean'].apply(lambda x: len(x.split()))
    df['cons_word_count'] = df['cons_clean'].apply(lambda x: len(x.split()))
    df['headline_word_count'] = df['headline_clean'].apply(lambda x: len(x.split()))
    df['total_word_count'] = df['pros_word_count'] + df['cons_word_count'] + df['headline_word_count']
    
    # Unique words
    df['pros_unique_words'] = df['pros_clean'].apply(lambda x: len(set(x.split())))
    df['cons_unique_words'] = df['cons_clean'].apply(lambda x: len(set(x.split())))
    
    # Vocabulary richness (unique/total ratio)
    df['pros_vocab_richness'] = df['pros_unique_words'] / (df['pros_word_count'] + 1)
    df['cons_vocab_richness'] = df['cons_unique_words'] / (df['cons_word_count'] + 1)
    
    # Ratios
    df['pros_cons_char_ratio'] = df['pros_char_length'] / (df['cons_char_length'] + 1)
    df['pros_cons_word_ratio'] = df['pros_word_count'] / (df['cons_word_count'] + 1)
    df['pros_proportion'] = df['pros_word_count'] / (df['total_word_count'] + 1)
    df['cons_proportion'] = df['cons_word_count'] / (df['total_word_count'] + 1)
    
    print("    - Linguistic patterns...")
    # Punctuation and patterns
    for col in ['pros', 'cons', 'headline']:
        patterns = df[col].apply(count_specific_patterns).apply(pd.Series)
        patterns.columns = [f'{col}_{c}' for c in patterns.columns]
        df = pd.concat([df, patterns], axis=1)
    
    print("    - Readability metrics...")
    # Readability
    pros_read = df['pros'].apply(compute_readability).apply(pd.Series)
    pros_read.columns = [f'pros_{c}' for c in pros_read.columns]
    cons_read = df['cons'].apply(compute_readability).apply(pd.Series)
    cons_read.columns = [f'cons_{c}' for c in cons_read.columns]
    df = pd.concat([df, pros_read, cons_read], axis=1)
    
    print("    - Keyword analysis...")
    # Keywords
    pros_kw = df['pros'].apply(count_keywords).apply(pd.Series)
    pros_kw.columns = [f'pros_{c}' for c in pros_kw.columns]
    cons_kw = df['cons'].apply(count_keywords).apply(pd.Series)
    cons_kw.columns = [f'cons_{c}' for c in cons_kw.columns]
    df = pd.concat([df, pros_kw, cons_kw], axis=1)
    
    if TEXTBLOB_AVAILABLE:
        print("    - Sentiment analysis...")
        # Sentiment
        pros_sent = df['pros'].apply(get_sentiment_features).apply(pd.Series)
        pros_sent.columns = [f'pros_{c}' for c in pros_sent.columns]
        cons_sent = df['cons'].apply(get_sentiment_features).apply(pd.Series)
        cons_sent.columns = [f'cons_{c}' for c in cons_sent.columns]
        df = pd.concat([df, pros_sent, cons_sent], axis=1)
        
        # Sentiment difference
        df['sentiment_diff'] = df['pros_polarity'] - df['cons_polarity']
        df['subjectivity_diff'] = df['pros_subjectivity'] - df['cons_subjectivity']
    
    # Year features
    df['year_normalized'] = (df['year_review'] - df['year_review'].min()) / (df['year_review'].max() - df['year_review'].min() + 1)
    df['is_recent'] = (df['year_review'] >= 2020).astype(int)
    
    return df

print("  Processing training data...")
train_df = engineer_features(train_df)
print("  Processing test data...")
test_df = engineer_features(test_df)

print(f"  Total engineered features: {len([c for c in train_df.columns if c not in ['rating', 'firm', 'job_title', 'pros', 'cons', 'headline', 'pros_clean', 'cons_clean', 'headline_clean', 'all_text']])}")

# ============================================================================
# SECTION 3: TF-IDF FEATURES (Optimized)
# ============================================================================
print("\n[3/10] CREATING TF-IDF FEATURES...")

# Pros TF-IDF (Optimized for best accuracy)
tfidf_pros = TfidfVectorizer(
    max_features=1200,   # Increased for better accuracy
    min_df=8,
    max_df=0.85,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\b[a-zA-Z]{2,}\b',
    stop_words='english',
    dtype=np.float32
)
X_train_pros_tfidf = tfidf_pros.fit_transform(train_df['pros_clean'])
X_test_pros_tfidf = tfidf_pros.transform(test_df['pros_clean'])
print(f"  Pros TF-IDF: {X_train_pros_tfidf.shape[1]} features")

# Cons TF-IDF
tfidf_cons = TfidfVectorizer(
    max_features=1200,   # Increased for better accuracy
    min_df=8,
    max_df=0.85,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\b[a-zA-Z]{2,}\b',
    stop_words='english',
    dtype=np.float32
)
X_train_cons_tfidf = tfidf_cons.fit_transform(train_df['cons_clean'])
X_test_cons_tfidf = tfidf_cons.transform(test_df['cons_clean'])
print(f"  Cons TF-IDF: {X_train_cons_tfidf.shape[1]} features")

# Headline TF-IDF
tfidf_headline = TfidfVectorizer(
    max_features=400,    # Increased from 300
    min_df=8,
    max_df=0.85,
    ngram_range=(1, 2),  # Added bigrams back
    sublinear_tf=True,
    stop_words='english',
    dtype=np.float32
)
X_train_headline_tfidf = tfidf_headline.fit_transform(train_df['headline_clean'])
X_test_headline_tfidf = tfidf_headline.transform(test_df['headline_clean'])
print(f"  Headline TF-IDF: {X_train_headline_tfidf.shape[1]} features")

# Character n-grams - Still disabled to prioritize word features
from scipy.sparse import csr_matrix
X_train_char = csr_matrix((X_train_pros_tfidf.shape[0], 0), dtype=np.float32)
X_test_char = csr_matrix((X_test_pros_tfidf.shape[0], 0), dtype=np.float32)
print(f"  Character n-grams: DISABLED (memory optimization)")

# ============================================================================
# SECTION 4: NUMERICAL FEATURES
# ============================================================================
print("\n[4/10] PREPARING NUMERICAL FEATURES...")

# Select all numerical features
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] 
                  and col != 'rating']
print(f"  Numerical features: {len(numerical_cols)}")

X_train_num = train_df[numerical_cols].values.astype(np.float32)
X_test_num = test_df[numerical_cols].values.astype(np.float32)

# Handle any NaN or inf values
X_train_num = np.nan_to_num(X_train_num, nan=0, posinf=0, neginf=0)
X_test_num = np.nan_to_num(X_test_num, nan=0, posinf=0, neginf=0)

# Scale numerical features
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num).astype(np.float32)
X_test_num_scaled = scaler.transform(X_test_num).astype(np.float32)

# ============================================================================
# SECTION 5: CATEGORICAL FEATURES
# ============================================================================
print("\n[5/10] ENCODING CATEGORICAL FEATURES...")

# Firm encoding with frequency
firm_counts = train_df['firm'].value_counts()
train_df['firm_frequency'] = train_df['firm'].map(firm_counts).fillna(1)
test_df['firm_frequency'] = test_df['firm'].map(firm_counts).fillna(1)

# Label encoding for firms
le_firm = LabelEncoder()
train_firm_encoded = le_firm.fit_transform(train_df['firm'].astype(str))

def safe_transform_firm(firm):
    if firm in le_firm.classes_:
        return le_firm.transform([firm])[0]
    return -1

test_firm_encoded = test_df['firm'].astype(str).apply(safe_transform_firm).values

X_train_firm = train_firm_encoded.reshape(-1, 1).astype(np.float32)
X_test_firm = test_firm_encoded.reshape(-1, 1).astype(np.float32)
print(f"  Unique firms in training: {len(le_firm.classes_)}")

# Add firm frequency
X_train_firm_freq = train_df['firm_frequency'].values.reshape(-1, 1).astype(np.float32)
X_test_firm_freq = test_df['firm_frequency'].values.reshape(-1, 1).astype(np.float32)

# ============================================================================
# SECTION 6: COMBINE ALL FEATURES FOR TRADITIONAL ML
# ============================================================================
print("\n[6/10] COMBINING FEATURES...")

X_train_sparse = hstack([
    X_train_pros_tfidf,
    X_train_cons_tfidf,
    X_train_headline_tfidf,
    X_train_char,
    csr_matrix(X_train_num_scaled),
    csr_matrix(X_train_firm),
    csr_matrix(X_train_firm_freq)
]).astype(np.float32)

X_test_sparse = hstack([
    X_test_pros_tfidf,
    X_test_cons_tfidf,
    X_test_headline_tfidf,
    X_test_char,
    csr_matrix(X_test_num_scaled),
    csr_matrix(X_test_firm),
    csr_matrix(X_test_firm_freq)
]).astype(np.float32)

y_train = train_df['rating'].values

print(f"  Total features for ML models: {X_train_sparse.shape[1]}")

# ============================================================================
# SECTION 7: DEEP LEARNING MODEL
# ============================================================================
print("\n[7/10] BUILDING DEEP LEARNING MODEL...")

# Prepare text sequences for neural network
MAX_WORDS = 20000
MAX_LEN = 150

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['all_text'])

X_train_seq = tokenizer.texts_to_sequences(train_df['all_text'])
X_test_seq = tokenizer.texts_to_sequences(test_df['all_text'])

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"  Vocabulary size: {min(len(tokenizer.word_index), MAX_WORDS)}")
print(f"  Sequence length: {MAX_LEN}")

# Convert labels to categorical (0-indexed for neural network)
y_train_nn = y_train - 1  # Convert 1-5 to 0-4

# Split for validation
X_tr_pad, X_val_pad, X_tr_num, X_val_num, y_tr_nn, y_val_nn = train_test_split(
    X_train_pad, X_train_num_scaled, y_train_nn, 
    test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_nn
)

# Build hybrid neural network (text + numerical features)
def build_hybrid_model():
    # Text input branch
    text_input = Input(shape=(MAX_LEN,), name='text_input')
    
    # Embedding layer
    embedding = Embedding(MAX_WORDS, 128, input_length=MAX_LEN)(text_input)
    
    # CNN for local patterns
    conv1 = Conv1D(128, 3, activation='relu', padding='same')(embedding)
    conv2 = Conv1D(128, 4, activation='relu', padding='same')(embedding)
    conv3 = Conv1D(128, 5, activation='relu', padding='same')(embedding)
    
    # Global pooling
    pool1 = GlobalMaxPooling1D()(conv1)
    pool2 = GlobalMaxPooling1D()(conv2)
    pool3 = GlobalMaxPooling1D()(conv3)
    
    # Concatenate CNN outputs
    text_features = Concatenate()([pool1, pool2, pool3])
    text_features = Dense(128, activation='relu')(text_features)
    text_features = Dropout(0.3)(text_features)
    text_features = BatchNormalization()(text_features)
    
    # Numerical input branch
    num_input = Input(shape=(X_train_num_scaled.shape[1],), name='num_input')
    num_features = Dense(64, activation='relu')(num_input)
    num_features = Dropout(0.3)(num_features)
    num_features = BatchNormalization()(num_features)
    num_features = Dense(32, activation='relu')(num_features)
    
    # Combine branches
    combined = Concatenate()([text_features, num_features])
    
    # Final dense layers
    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    
    # Output layer (5 classes)
    output = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=[text_input, num_input], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("  Building hybrid CNN model...")
nn_model = build_hybrid_model()
nn_model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

print("\n  Training neural network...")
history = nn_model.fit(
    [X_tr_pad, X_tr_num],
    y_tr_nn,
    validation_data=([X_val_pad, X_val_num], y_val_nn),
    epochs=20,           # Reduced from 30
    batch_size=128,      # Increased from 64 for stability
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Neural network predictions
nn_train_proba = nn_model.predict([X_train_pad, X_train_num_scaled], verbose=0)
nn_test_proba = nn_model.predict([X_test_pad, X_test_num_scaled], verbose=0)

nn_train_pred = np.argmax(nn_train_proba, axis=1) + 1  # Convert back to 1-5
nn_test_pred = np.argmax(nn_test_proba, axis=1) + 1

nn_train_accuracy = accuracy_score(y_train, nn_train_pred) * 100
print(f"\n  Neural Network Training Accuracy: {nn_train_accuracy:.2f}%")

# ============================================================================
# SECTION 8: GRADIENT BOOSTING MODELS
# ============================================================================
print("\n[8/10] TRAINING GRADIENT BOOSTING MODELS...")

# Split for validation - convert labels to 0-indexed for gradient boosting
y_train_gb = y_train - 1  # Convert 1-5 to 0-4 for XGBoost/LightGBM
X_tr_sp, X_val_sp, y_tr, y_val = train_test_split(
    X_train_sparse, y_train_gb, test_size=0.15,
    random_state=RANDOM_STATE, stratify=y_train_gb
)

models = {}
predictions = {}

# XGBoost - DISABLED to save memory for Neural Network
# if XGBOOST_AVAILABLE:
print("  XGBoost: DISABLED (memory optimization for NN)")
XGBOOST_AVAILABLE = False
xgb_accuracy = 0

# LightGBM - DISABLED to save memory for Neural Network
# if LIGHTGBM_AVAILABLE:
print("  LightGBM: DISABLED (memory optimization for NN)")
LIGHTGBM_AVAILABLE = False
lgb_accuracy = 0

# ============================================================================
# SECTION 9: ENSEMBLE PREDICTIONS
# ============================================================================
print("\n[9/10] CREATING FINAL PREDICTIONS...")

# Since XGBoost and LightGBM are disabled, use Neural Network only
print("  Using Neural Network predictions (XGBoost/LightGBM disabled)")

# Final predictions come directly from Neural Network
final_predictions = nn_test_pred
final_train_predictions = nn_train_pred
final_accuracy = nn_train_accuracy

print(f"\n  Model Results:")
print(f"    Neural Network: {nn_train_accuracy:.2f}%")
print(f"    (XGBoost: DISABLED)")
print(f"    (LightGBM: DISABLED)")
print(f"\n  Final Model: Neural Network with {final_accuracy:.2f}% training accuracy")
algorithm_name = "Deep Neural Network (CNN)"

print(f"\n  SELECTED MODEL: {algorithm_name}")
print(f"  Final Training Accuracy: {final_accuracy:.2f}%")

# ============================================================================
# SECTION 10: SAVE PREDICTIONS
# ============================================================================
print("\n[10/10] SAVING PREDICTIONS...")

pd.DataFrame({"predicted_rating": final_predictions.astype(int)}).to_csv(
    os.path.join(ROOT, "predictions.csv"), index=False)
print("  Saved: predictions.csv")

print(f"\n  Prediction Distribution:")
for rating in range(1, 6):
    count = np.sum(final_predictions == rating)
    pct = count / len(final_predictions) * 100
    print(f"    Rating {rating}: {count:6d} ({pct:.2f}%)")

# ============================================================================
# SECTION 11: GENERATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS FOR Q1b...")
print("=" * 70)

plt.style.use('seaborn-v0_8-darkgrid')

# 11.1 Word Clouds
if WORDCLOUD_AVAILABLE:
    print("  Creating word clouds...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    text_5star = ' '.join(train_df[train_df['rating'] == 5]['all_text'].values)
    text_1star = ' '.join(train_df[train_df['rating'] == 1]['all_text'].values)
    
    wc_5star = WordCloud(width=800, height=400, background_color='white',
                         colormap='Greens', max_words=150).generate(text_5star)
    axes[0].imshow(wc_5star, interpolation='bilinear')
    axes[0].set_title('5-Star Reviews', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    wc_1star = WordCloud(width=800, height=400, background_color='white',
                         colormap='Reds', max_words=150).generate(text_1star)
    axes[1].imshow(wc_1star, interpolation='bilinear')
    axes[1].set_title('1-Star Reviews', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('Word Cloud: 5-Star vs 1-Star Reviews', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_wordclouds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved: fig1_wordclouds.png")

# 11.3 Distribution Comparison
print("  Creating distribution comparison...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

features_to_compare = [
    ('year_review', 'Year of Review'),
    ('pros_char_length', 'Pros Character Length'),
    ('cons_char_length', 'Cons Character Length'),
    ('pros_cons_word_ratio', 'Pros/Cons Word Ratio'),
    ('total_word_count', 'Total Word Count'),
    ('pros_vocab_richness', 'Pros Vocabulary Richness')
]

for idx, (feat, title) in enumerate(features_to_compare):
    ax = axes[idx]
    ax.hist(train_df[feat], bins=50, alpha=0.5, label='Training', density=True, color='blue')
    ax.hist(test_df[feat], bins=50, alpha=0.5, label='Test', density=True, color='red')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend()

plt.suptitle('Feature Distribution: Training vs Test Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_train_test_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig2_train_test_distribution.png")

# 11.4 Confusion Matrix
print("  Creating confusion matrix...")
cm = confusion_matrix(y_train, final_train_predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
axes[0].set_xlabel('Predicted Rating', fontsize=12)
axes[0].set_ylabel('Actual Rating', fontsize=12)
axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')

sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
axes[1].set_xlabel('Predicted Rating', fontsize=12)
axes[1].set_ylabel('Actual Rating', fontsize=12)
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')

plt.suptitle(f'Model Performance: {algorithm_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig3_confusion_matrix.png")

# 11.5 Error Analysis
print("  Creating error analysis...")
errors = final_train_predictions - y_train

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Actual vs Predicted distribution
ax = axes[0, 0]
x = np.arange(1, 6)
width = 0.35
actual_counts = [np.sum(y_train == i) for i in range(1, 6)]
pred_counts = [np.sum(final_train_predictions == i) for i in range(1, 6)]
ax.bar(x - width/2, actual_counts, width, label='Actual (y)', color='steelblue')
ax.bar(x + width/2, pred_counts, width, label='Predicted (ŷ)', color='coral')
ax.set_xlabel('Rating', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution: Actual vs Predicted', fontsize=12, fontweight='bold')
ax.legend()

# Error distribution
ax = axes[0, 1]
error_counts = pd.Series(errors).value_counts().sort_index()
ax.bar(error_counts.index, error_counts.values, color='purple', alpha=0.7)
ax.set_xlabel('Prediction Error (ŷ - y)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)

# Error by rating
ax = axes[1, 0]
error_by_rating = [np.abs(errors[y_train == i]) for i in range(1, 6)]
bp = ax.boxplot(error_by_rating, labels=[1, 2, 3, 4, 5], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
ax.set_xlabel('Actual Rating', fontsize=11)
ax.set_ylabel('Absolute Error', fontsize=11)
ax.set_title('Error Magnitude by Actual Rating', fontsize=12, fontweight='bold')

# Accuracy by rating
ax = axes[1, 1]
accuracy_by_rating = [np.mean(final_train_predictions[y_train == i] == i) * 100 for i in range(1, 6)]
ax.bar(range(1, 6), accuracy_by_rating, color='teal', alpha=0.8)
ax.set_xlabel('Rating', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Classification Accuracy by Rating', fontsize=12, fontweight='bold')
for i, acc in enumerate(accuracy_by_rating):
    ax.text(i+1, acc + 1, f'{acc:.1f}%', ha='center', fontweight='bold')

plt.suptitle('Prediction Error Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_error_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig4_error_analysis.png")

# 11.6 Actionable Insights
print("  Creating actionable insights visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Sentiment by rating (if available)
if TEXTBLOB_AVAILABLE:
    ax = axes[0, 0]
    pros_polarity = train_df.groupby('rating')['pros_polarity'].mean()
    cons_polarity = train_df.groupby('rating')['cons_polarity'].mean()
    x = np.arange(1, 6)
    width = 0.35
    ax.bar(x - width/2, pros_polarity, width, label='Pros Sentiment', color='green', alpha=0.7)
    ax.bar(x + width/2, cons_polarity, width, label='Cons Sentiment', color='red', alpha=0.7)
    ax.set_xlabel('Rating', fontsize=11)
    ax.set_ylabel('Average Polarity', fontsize=11)
    ax.set_title('Sentiment Polarity by Rating', fontsize=12, fontweight='bold')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
else:
    ax = axes[0, 0]
    train_df.groupby('rating')['pros_cons_word_ratio'].mean().plot(kind='bar', ax=ax, color='teal')
    ax.set_title('Pros/Cons Word Ratio by Rating', fontsize=12, fontweight='bold')

# Keyword analysis
ax = axes[0, 1]
pos_kw = train_df.groupby('rating')['pros_positive_keyword_count'].mean()
neg_kw = train_df.groupby('rating')['cons_negative_keyword_count'].mean()
x = np.arange(1, 6)
ax.plot(x, pos_kw, marker='o', label='Positive Keywords (Pros)', color='green', linewidth=2)
ax.plot(x, neg_kw, marker='s', label='Negative Keywords (Cons)', color='red', linewidth=2)
ax.set_xlabel('Rating', fontsize=11)
ax.set_ylabel('Average Keyword Count', fontsize=11)
ax.set_title('Keyword Usage by Rating', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Word count patterns
ax = axes[1, 0]
pros_wc = train_df.groupby('rating')['pros_word_count'].mean()
cons_wc = train_df.groupby('rating')['cons_word_count'].mean()
x = np.arange(1, 6)
width = 0.35
ax.bar(x - width/2, pros_wc, width, label='Pros', color='green', alpha=0.7)
ax.bar(x + width/2, cons_wc, width, label='Cons', color='red', alpha=0.7)
ax.set_xlabel('Rating', fontsize=11)
ax.set_ylabel('Average Word Count', fontsize=11)
ax.set_title('Average Word Count by Rating', fontsize=12, fontweight='bold')
ax.legend()

axes[1, 1].axis('off')  # findings are summarised in the README

plt.suptitle('Actionable Insights: Drivers of Employer Ratings', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_rating_drivers.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig5_rating_drivers.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Format model accuracies safely
try:
    xgb_display = f"{float(xgb_accuracy):.2f}%" if XGBOOST_AVAILABLE else "DISABLED"
except:
    xgb_display = "DISABLED"

try:
    lgb_display = f"{float(lgb_accuracy):.2f}%" if LIGHTGBM_AVAILABLE else "DISABLED"
except:
    lgb_display = "DISABLED"

print(f"""
MODEL PERFORMANCE:
  • Neural Network:  {nn_train_accuracy:.2f}%
  • XGBoost:         {xgb_display}
  • LightGBM:        {lgb_display}
  • Final Model:     {algorithm_name}

SELECTED MODEL: {algorithm_name}
FINAL ACCURACY: {final_accuracy:.2f}%

FEATURE ENGINEERING SUMMARY:
  • TF-IDF Features: {X_train_pros_tfidf.shape[1] + X_train_cons_tfidf.shape[1] + X_train_headline_tfidf.shape[1]}
  • Character N-grams: {X_train_char.shape[1]}
  • Numerical Features: {len(numerical_cols)}
  • Total Features: {X_train_sparse.shape[1]}

FILES GENERATED:
  predictions.csv and figures/fig1-fig5
""")

print("=" * 70)
print("Q1 COMPLETE - ADVANCED DEEP LEARNING MODEL")
print("=" * 70)
