import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Configuration
MODEL_DIR = "./heart_sound_models"
MODEL_PATH = os.path.join(MODEL_DIR, "heart_sound_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

st.set_page_config(page_title="Heart Sound Classifier", page_icon="❤️")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None, None
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else None
        return model, le, scaler, metrics
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def extract_features(audio_data, sr):
    """
    Extract MFCC features, Deltas, Spectral features, and ZCR from audio data.
    Must match the logic in model_training.py
    """
    try:
        # Ensure we have up to 10 seconds
        if len(audio_data) > 10 * sr:
            audio_data = audio_data[:10 * sr]
        
        y = audio_data
        
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
        st.error(f"Error processing audio: {e}")
        return None

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def main():
    st.title("❤️ Heart Sound Classifier (Enhanced)")
    
    model, le, scaler, metrics = load_model()
    
    if model is None:
        st.warning("Model files not found. Please ensure `heart_sound_model.pkl`, `label_encoder.pkl`, and `scaler.pkl` are in `heart_sound_models/`.")
        return

    # Sidebar for Model Performance
    st.sidebar.title("Model Performance")
    show_metrics = st.sidebar.checkbox("Show Model Metrics", value=False)
    
    if show_metrics and metrics:
        st.sidebar.markdown(f"**Test Accuracy:** `{metrics['accuracy']:.2%}`")
        
        st.subheader("📊 Model Performance Metrics")
        
        # Metrics Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Confusion Matrix")
            fig_cm = plot_confusion_matrix(metrics['confusion_matrix'], metrics['classes'])
            st.pyplot(fig_cm)
            
        with col2:
            st.write("### Classification Report")
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))

        st.write("---")

    st.write("Upload a heart sound recording (.wav) to classify it as **Normal** or **Abnormal**.")

    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Heart Sound"):
            with st.spinner("Analyzing..."):
                # Load audio using librosa
                # Librosa expects a file path or file-like object
                # IMPORTANT: We must use the same sampling rate as training (default 22050)
                # Removing sr=None to use default 22050 Hz.
                y, sr = librosa.load(uploaded_file)
                
                # Plot waveform
                fig, ax = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                st.pyplot(fig)

                # Extract features
                features = extract_features(y, sr)
                
                if features is not None:
                    # Reshape for prediction
                    features = features.reshape(1, -1)
                    
                    # Scale features (CRITICAL!)
                    features_scaled = scaler.transform(features)
                    
                    # Predict
                    prediction_idx = model.predict(features_scaled)[0]
                    prediction_label = le.inverse_transform([prediction_idx])[0]
                    prediction_prob = model.predict_proba(features_scaled)[0]
                    
                    # Display results
                    confidence = prediction_prob[prediction_idx]
                    
                    if prediction_label == "Normal":
                        st.success(f"### Prediction: {prediction_label} Heart Sound")
                    else:
                        st.error(f"### Prediction: {prediction_label} Heart Sound")
                        
                    st.write(f"Confidence: **{confidence:.2%}**")
                    
                    # Detailed probabilities
                    st.write("---")
                    st.write("Probability Distribution:")
                    probs = {cls: prob for cls, prob in zip(le.classes_, prediction_prob)}
                    st.json(probs)

if __name__ == "__main__":
    main()
