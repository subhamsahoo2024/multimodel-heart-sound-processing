import os
import glob
import pandas as pd
import numpy as np
import librosa
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuration
DATA_DIR = r"d:\ml practice\mutimodel-heart-sounds\pcg_data2"
CSV_PATH = os.path.join(DATA_DIR, "training_data.csv")
AUDIO_DIR = os.path.join(DATA_DIR, "training_data", "training_data")

# Output Directory
OUTPUT_DIR = "heart_sound_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "heart_sound_model.pkl")
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.pkl")

FEATURES_CACHE = "X_features.npy"
LABELS_CACHE = "y_labels.npy"

def extract_features(file_path):
    """
    Extract MFCC features, Deltas, Spectral features, and ZCR from an audio file.
    """
    try:
        y, sr = librosa.load(file_path, duration=10) # Load up to 10 seconds
        
        # 1. MFCCs (Mean & Variance)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_var = np.var(mfccs.T, axis=0)
        
        # 2. Delta MFCCs (Temporal changes)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta_mfccs.T, axis=0)
        delta_var = np.var(delta_mfccs.T, axis=0)
        
        # 3. Spectral Features (Timbre/Brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(spectral_centroid)
        cent_var = np.var(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bw_mean = np.mean(spectral_bandwidth)
        spec_bw_var = np.var(spectral_bandwidth)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_var = np.var(spectral_rolloff)
        
        # 4. Zero Crossing Rate (Noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        features = np.concatenate([
            mfccs_mean, mfccs_var,
            delta_mean, delta_var,
            [cent_mean, cent_var, spec_bw_mean, spec_bw_var, rolloff_mean, rolloff_var, zcr_mean, zcr_var]
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_patient(row):
    patient_id = str(row['Patient ID'])
    label = row['Label']
    
    # Look for all audio files for this patient
    pattern = os.path.join(AUDIO_DIR, f"{patient_id}_*.wav")
    audio_files = glob.glob(pattern)
    
    patient_features = []
    patient_labels = []
    
    for file_path in audio_files:
        feat = extract_features(file_path)
        if feat is not None:
            patient_features.append(feat)
            patient_labels.append(label)
            
    return patient_features, patient_labels

def load_data():
    """
    Load data from CSV and audio files. Uses caching to speed up subsequent runs.
    """
    if os.path.exists(FEATURES_CACHE) and os.path.exists(LABELS_CACHE):
        print("Loading features from cache...")
        return np.load(FEATURES_CACHE), np.load(LABELS_CACHE)

    print("Loading CSV data...")
    df = pd.read_csv(CSV_PATH)
    
    # Filter valid labels
    df = df[df['Murmur'] != 'Unknown']
    
    # Map labels: Absent -> Normal, Present -> Abnormal
    df['Label'] = df['Murmur'].map({'Absent': 'Normal', 'Present': 'Abnormal'})
    
    print(f"Processing {len(df)} patients with Parallel processing...")
    
    # Run parallel processing
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_patient)(row) for index, row in df.iterrows()
    )
    
    # Flatten results
    features = []
    labels = []
    for res_feats, res_labels in results:
        features.extend(res_feats)
        labels.extend(res_labels)
            
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Saving features to cache ({len(X)} samples)...")
    np.save(FEATURES_CACHE, X)
    np.save(LABELS_CACHE, y)
    
    return X, y

def train_model():
    X, y = load_data()
    
    print(f"Total samples: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Train model
    print("Training Gradient Boosting Classifier...")
    # Gradient Boosting handles imbalance implicitly well, but we can verify.
    # Unfortunately GBC doesn't support class_weight directly in older sklearn versions, but we can try sample_weight if needed.
    # However, let's try standard GBC first.
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=le.classes_)
    print("Classification Report:\n", report_str)
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save model and scaler
    print("Saving model, scaler, and metrics...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save metrics for visualization
    metrics = {
        'accuracy': acc,
        'classification_report': report_dict,
        'confusion_matrix': cm,
        'classes': le.classes_,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    joblib.dump(metrics, METRICS_PATH)
    
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    train_model()
